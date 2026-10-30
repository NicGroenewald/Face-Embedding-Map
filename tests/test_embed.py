"""Tests for embedding-cache provenance in ``fem.embed``.

Why these tests exist
---------------------
The Olivetti embedding cache is a shortcut that downstream analysis trusts.  A
stale cache (different checkpoint, different preprocessing, changed dataset, or
a corrupt matrix) would silently change every Map B number.  These tests use a
tiny synthetic dataset and a fake embedder, so they run in milliseconds and
never download weights or touch ``results/``.

The policy under test is: a cache is reused only when all provenance fields
match *and* the stored rows are finite unit vectors whose norm metadata is
consistent; otherwise ``embed_olivetti`` warns and rebuilds.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem import embed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fast_checkpoint(monkeypatch):
    """Hash of a 107 MB checkpoint is irrelevant for cache-rule tests."""
    monkeypatch.setattr(embed, "checkpoint_sha256", lambda: "checkpoint-test-sha")


def make_dataset(n_images: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = rng.random((n_images, 64, 64), dtype=np.float32)
    labels = np.repeat(np.arange(n_images // 2), 2)
    return images, labels


def unit_embeddings(n_images: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n_images, embed.EMBED_DIM)).astype(np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def write_cache(path, embeddings: np.ndarray, provenance: dict) -> None:
    """Write a cache in the current format (valid unless inputs are bad)."""
    np.savez_compressed(
        path,
        embeddings=embeddings,
        **provenance,
        **embed._embedding_integrity(embeddings),
    )


class FakeEmbedder:
    """Instant stand-in for the frozen model; records how often it was called."""

    calls = 0

    def __init__(self, device=None):
        self.device = device or "cpu"

    def embed_images(self, images, batch_size=32):
        type(self).calls += 1
        return unit_embeddings(n_images=len(images), seed=type(self).calls)


class BadEmbedder:
    """Returns non-unit rows, which a cache must never store."""

    def __init__(self, device=None):
        self.device = device or "cpu"

    def embed_images(self, images, batch_size=32):
        return np.ones((len(images), embed.EMBED_DIM), dtype=np.float32) * 2.0


@pytest.fixture()
def loaded_dataset(monkeypatch):
    images, labels = make_dataset()
    monkeypatch.setattr(embed, "load_olivetti", lambda: (images, labels))
    monkeypatch.setattr(embed, "FaceEmbedder", FakeEmbedder)
    FakeEmbedder.calls = 0
    return images, labels


# ---------------------------------------------------------------------------
# fingerprints / provenance
# ---------------------------------------------------------------------------
def test_dataset_fingerprint_is_deterministic_and_sensitive():
    images, labels = make_dataset()
    fingerprint = embed.dataset_fingerprint(images, labels)

    assert fingerprint == embed.dataset_fingerprint(images.copy(), labels.copy())
    assert len(fingerprint) == 64

    changed_pixels = images.copy()
    changed_pixels[0, 0, 0] = np.float32(changed_pixels[0, 0, 0] + 1e-3)
    assert embed.dataset_fingerprint(changed_pixels, labels) != fingerprint

    changed_labels = labels.copy()
    changed_labels[0] = 1
    assert embed.dataset_fingerprint(images, changed_labels) != fingerprint


def test_preprocess_fingerprint_is_deterministic():
    first = embed.preprocess_fingerprint()
    assert first == embed.preprocess_fingerprint()
    assert len(first) == 64


def test_cache_provenance_has_all_required_fields():
    images, labels = make_dataset()
    provenance = embed.cache_provenance(images, labels)

    assert set(embed._PROVENANCE_KEYS) == set(provenance)
    assert provenance["cache_format"] == embed.CACHE_FORMAT_VERSION
    assert provenance["model"] == embed.MODEL_NAME
    assert provenance["checkpoint_sha256"] == "checkpoint-test-sha"
    assert provenance["dataset_sha256"] == embed.dataset_fingerprint(images, labels)
    assert provenance["preprocess_sha256"] == embed.preprocess_fingerprint()
    assert provenance["seed"] == 0
    assert provenance["embed_dim"] == 512
    assert provenance["face_size"] == 160
    assert provenance["n_images"] == 8


def test_cache_mismatches_reports_changed_and_missing_fields():
    images, labels = make_dataset()
    expected = embed.cache_provenance(images, labels)
    assert embed._cache_mismatches(expected, expected) == []

    changed = dict(expected)
    changed["checkpoint_sha256"] = "some-other-checkpoint"
    reasons = embed._cache_mismatches(changed, expected)
    assert any("checkpoint_sha256" in reason for reason in reasons)

    missing = {key: value for key, value in expected.items() if key != "dataset_sha256"}
    reasons = embed._cache_mismatches(missing, expected)
    assert any("dataset_sha256" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# embedding validation
# ---------------------------------------------------------------------------
def test_validate_embeddings_accepts_unit_finite_rows():
    assert embed._validate_embeddings(unit_embeddings(), 8) == []


def test_validate_embeddings_rejects_bad_matrices():
    assert any("shape" in problem for problem in embed._validate_embeddings(np.zeros((8, 10))))
    assert any("rows" in problem for problem in embed._validate_embeddings(unit_embeddings(7), 8))

    non_finite = unit_embeddings()
    non_finite[0, 0] = np.nan
    assert any("non-finite" in problem for problem in embed._validate_embeddings(non_finite))

    assert any("row norms" in problem for problem in embed._validate_embeddings(unit_embeddings() * 2))


# ---------------------------------------------------------------------------
# cache reuse / rebuild policy
# ---------------------------------------------------------------------------
def test_embed_olivetti_writes_provenance_and_reuses_valid_cache(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"

    first = embed.embed_olivetti(cache=cache_path)
    assert first.shape == (8, embed.EMBED_DIM)
    assert FakeEmbedder.calls == 1
    assert cache_path.exists()

    expected = embed.cache_provenance(images, labels)
    with np.load(cache_path) as cached:
        assert set(embed._PROVENANCE_KEYS) | set(embed._INTEGRITY_KEYS) <= set(cached.files)
        assert str(cached["dataset_sha256"].item()) == expected["dataset_sha256"]
        assert str(cached["checkpoint_sha256"].item()) == "checkpoint-test-sha"
        assert float(cached["row_norm_min"]) == pytest.approx(1.0, abs=1e-5)

    second = embed.embed_olivetti(cache=cache_path)
    np.testing.assert_array_equal(first, second)
    assert FakeEmbedder.calls == 1  # no rebuild: provenance matched


def test_embed_olivetti_rebuilds_legacy_cache(tmp_path, loaded_dataset):
    cache_path = tmp_path / "embeddings_olivetti.npz"
    # The previous cache format: only model/seed/sizes, no provenance or
    # integrity metadata.  It must not be silently accepted.
    np.savez_compressed(
        cache_path,
        embeddings=unit_embeddings(),
        model=embed.MODEL_NAME,
        seed=np.int64(0),
        embed_dim=np.int64(embed.EMBED_DIM),
        face_size=np.int64(embed.FACE_SIZE),
    )

    with pytest.warns(UserWarning, match="rebuilding"):
        result = embed.embed_olivetti(cache=cache_path)

    assert result.shape == (8, embed.EMBED_DIM)
    assert FakeEmbedder.calls == 1
    with np.load(cache_path) as cached:
        assert "dataset_sha256" in cached.files


def test_embed_olivetti_rebuilds_when_dataset_changes(tmp_path, loaded_dataset, monkeypatch):
    _images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    embed.embed_olivetti(cache=cache_path)
    assert FakeEmbedder.calls == 1

    other_images, _ = make_dataset(seed=1)
    monkeypatch.setattr(embed, "load_olivetti", lambda: (other_images, labels))
    with pytest.warns(UserWarning, match="dataset_sha256"):
        embed.embed_olivetti(cache=cache_path)

    assert FakeEmbedder.calls == 2  # rebuilt for the new dataset


def test_embed_olivetti_rebuilds_when_checkpoint_changes(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    embed.embed_olivetti(cache=cache_path)

    # Change the *expected* checkpoint hash behind the cache's back.
    provenance = embed.cache_provenance(images, labels)
    provenance["checkpoint_sha256"] = "a-different-checkpoint"
    write_cache(cache_path, unit_embeddings(seed=2), provenance)

    with pytest.warns(UserWarning, match="checkpoint_sha256"):
        embed.embed_olivetti(cache=cache_path)
    assert FakeEmbedder.calls == 2


def test_embed_olivetti_rebuilds_cache_with_nonunit_rows(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    bad = unit_embeddings() * 2.0
    write_cache(cache_path, bad, embed.cache_provenance(images, labels))

    with pytest.warns(UserWarning, match="row norms"):
        embed.embed_olivetti(cache=cache_path)
    assert FakeEmbedder.calls == 1


def test_embed_olivetti_rebuilds_cache_with_nonfinite_rows(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    bad = unit_embeddings()
    bad[0, 0] = np.nan
    write_cache(cache_path, bad, embed.cache_provenance(images, labels))

    with pytest.warns(UserWarning, match="non-finite"):
        embed.embed_olivetti(cache=cache_path)
    assert FakeEmbedder.calls == 1


def test_embed_olivetti_rebuilds_cache_with_inconsistent_norm_metadata(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    embeddings = unit_embeddings()
    provenance = embed.cache_provenance(images, labels)
    integrity = embed._embedding_integrity(embeddings)
    integrity["row_norm_max"] = 0.5  # tampered
    np.savez_compressed(cache_path, embeddings=embeddings, **provenance, **integrity)

    with pytest.warns(UserWarning, match="row_norm_max"):
        embed.embed_olivetti(cache=cache_path)
    assert FakeEmbedder.calls == 1


def test_embed_olivetti_rebuilds_unreadable_cache(tmp_path, loaded_dataset):
    cache_path = tmp_path / "embeddings_olivetti.npz"
    cache_path.write_bytes(b"this is not an npz file")

    with pytest.warns(UserWarning, match="unreadable cache"):
        embed.embed_olivetti(cache=cache_path)
    assert FakeEmbedder.calls == 1


def test_embed_olivetti_refuses_to_cache_invalid_embeddings(tmp_path, monkeypatch):
    images, labels = make_dataset()
    monkeypatch.setattr(embed, "load_olivetti", lambda: (images, labels))
    monkeypatch.setattr(embed, "FaceEmbedder", BadEmbedder)
    cache_path = tmp_path / "embeddings_olivetti.npz"

    with pytest.raises(RuntimeError, match="invalid embedding matrix"):
        embed.embed_olivetti(cache=cache_path)
    assert not cache_path.exists()


# ---------------------------------------------------------------------------
# read-only loader used by the analysis runners
# ---------------------------------------------------------------------------
def test_load_cached_olivetti_embeddings_returns_valid_cache(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    embeddings = unit_embeddings()
    write_cache(cache_path, embeddings, embed.cache_provenance(images, labels))

    loaded = embed.load_cached_olivetti_embeddings(cache=cache_path)

    np.testing.assert_array_equal(loaded, embeddings)
    assert FakeEmbedder.calls == 0  # the loader must never construct the model


def test_load_cached_olivetti_embeddings_missing_cache_is_actionable(tmp_path, loaded_dataset):
    with pytest.raises(FileNotFoundError, match="make maps"):
        embed.load_cached_olivetti_embeddings(cache=tmp_path / "missing.npz")


def test_load_cached_olivetti_embeddings_rejects_stale_cache(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    provenance = embed.cache_provenance(images, labels)
    provenance["dataset_sha256"] = "a-different-dataset"
    write_cache(cache_path, unit_embeddings(), provenance)

    with pytest.raises(ValueError, match="make maps") as excinfo:
        embed.load_cached_olivetti_embeddings(cache=cache_path)
    assert "dataset_sha256" in str(excinfo.value)
    assert FakeEmbedder.calls == 0


def test_load_cached_olivetti_embeddings_rejects_legacy_cache(tmp_path, loaded_dataset):
    cache_path = tmp_path / "embeddings_olivetti.npz"
    np.savez_compressed(
        cache_path,
        embeddings=unit_embeddings(),
        model=embed.MODEL_NAME,
        seed=np.int64(0),
        embed_dim=np.int64(embed.EMBED_DIM),
        face_size=np.int64(embed.FACE_SIZE),
    )

    with pytest.raises(ValueError, match="stale or invalid"):
        embed.load_cached_olivetti_embeddings(cache=cache_path)


def test_load_cached_olivetti_embeddings_rejects_bad_rows(tmp_path, loaded_dataset):
    images, labels = loaded_dataset
    cache_path = tmp_path / "embeddings_olivetti.npz"
    write_cache(cache_path, unit_embeddings() * 2.0, embed.cache_provenance(images, labels))

    with pytest.raises(ValueError, match="row norms"):
        embed.load_cached_olivetti_embeddings(cache=cache_path)
