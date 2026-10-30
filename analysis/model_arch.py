"""E5 — verified InceptionResnetV1(vggface2) architecture dump -> results/model_arch.json.

Why this exists
---------------
The paper's model diagram (``paper/figures/fig_model.pdf``) has to show real
tensor shapes.  Guessing them from a blog post is how papers end up with a
diagram that does not match the code.  Worse, facenet-pytorch's implementation
of "InceptionResnetV1" has changed over releases: this version (2.5.3) has
``conv2d_4b`` / ``mixed_6a`` / ``mixed_7a`` / ``block8`` and drops the old
``maxpool_5a`` naming.  So the only trustworthy source of shapes is a forward
pass with hooks on the checkpoint we actually embed with.

What it records
---------------
For every top-level module in forward order: name, module type, output shape,
parameter count.  Plus the repeat-block counts (5x Block35, 10x Block17,
5x Block8 + one final Block8), the total embedding parameter count, and an
invariant check that the output is L2-normalised (this is what the whole
"distances are Euclidean between unit vectors" analysis relies on).

Run: ``$PY analysis/model_arch.py``   (torch only; no fem dependency)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis._common import (  # noqa: E402  (path bootstrap must run first)
    RESULTS_DIR,
    common_header,
    describe_device,
    save_json,
)

FACE_SIZE = 160
EMBED_DIM = 512
CHECKPOINT = Path.home() / ".cache/torch/checkpoints/20180402-114759-vggface2.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def main() -> dict:
    from facenet_pytorch import InceptionResnetV1

    device = describe_device()
    print(f"[model_arch] device={device}")

    # classify=False asks for the 512-d embedding head, not the 8631-class logits.
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)

    # One fixed batch. Shapes do not depend on the values, but randn avoids any
    # degenerate all-zero path and makes the run deterministic across machines.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, FACE_SIZE, FACE_SIZE, generator=g).to(device)

    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach()
        return hook

    handles = []
    for name, child in model.named_children():
        handles.append(child.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        out = model(x)

    for h in handles:
        h.remove()

    # --- invariants the analysis depends on -------------------------------
    norms = out.norm(dim=1)
    assert out.shape == (4, EMBED_DIM), f"unexpected embedding shape {tuple(out.shape)}"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"model output is not unit-norm: min={norms.min():.6f} max={norms.max():.6f}"
    )

    # --- per-layer dump ----------------------------------------------------
    layers = []
    unused = []
    for idx, (name, child) in enumerate(model.named_children()):
        params = _module_params(child)
        if name in captured:
            shape = list(captured[name].shape)
            # Shape after the module's own last op; for Sequential blocks the
            # hook fires on the wrapper, which is exactly what the diagram wants.
            layers.append(
                {
                    "index": idx,
                    "name": name,
                    "type": type(child).__name__,
                    "output_shape": shape,
                    "params": params,
                    "runs_in_forward": True,
                }
            )
        else:
            unused.append(
                {
                    "index": idx,
                    "name": name,
                    "type": type(child).__name__,
                    "params": params,
                    "runs_in_forward": False,
                    "reason": "classifier head kept for state-dict compatibility; "
                    "not executed when classify=False",
                }
            )

    # --- repeat-block anatomy (verified by len(), not by reading the paper) ---
    blocks = {}
    for name in ("repeat_1", "repeat_2", "repeat_3"):
        seq = getattr(model, name)
        first = seq[0]
        blocks[name] = {
            "count": len(seq),
            "block_type": type(first).__name__,
            "scale": float(getattr(first, "scale", float("nan"))),
            "output_shape": list(captured[name].shape),
            "params": _module_params(seq),
        }
    final_block = model.block8
    blocks["block8"] = {
        "count": 1,
        "block_type": type(final_block).__name__,
        "scale": float(getattr(final_block, "scale", float("nan"))),
        "output_shape": list(captured["block8"].shape),
        "params": _module_params(final_block),
    }

    # --- parameter accounting ---------------------------------------------
    total_model_params = int(sum(p.numel() for p in model.parameters()))
    classifier_params = int(sum(p.numel() for p in model.logits.parameters()))
    embedding_params = total_model_params - classifier_params

    checkpoint_info = {"path": str(CHECKPOINT), "exists": CHECKPOINT.exists()}
    if CHECKPOINT.exists():
        checkpoint_info["bytes"] = CHECKPOINT.stat().st_size
        checkpoint_info["sha256"] = _sha256(CHECKPOINT)

    payload = {
        "common": common_header(
            device,
            notes=[
                "Shapes measured with forward hooks on the actual vggface2 checkpoint, "
                "not copied from a diagram.",
                "FLOPs are deliberately not reported: hooks do not measure them and "
                "guessing them would be inventing numbers.",
                "The 'logits' classifier head ships in the checkpoint state dict but is "
                "not executed when classify=False, so it is excluded from embedding_params.",
            ],
        ),
        "model": {
            "name": "InceptionResnetV1",
            "pretrained": "vggface2",
            "classify": False,
            "checkpoint": checkpoint_info,
            "input_shape": [1, 3, FACE_SIZE, FACE_SIZE],
            "input_pixels": "RGB, standardised (x*255 - 127.5) / 128 on a 0..1 input",
            "output_dim": EMBED_DIM,
            "output_shape": list(out.shape),
            "output_l2_normalised": True,
            "output_norm": {
                "min": float(norms.min()),
                "mean": float(norms.mean()),
                "max": float(norms.max()),
            },
            "eval_mode": True,
            "parameters_requires_grad": all(not p.requires_grad for p in model.parameters()),
            "total_model_params": total_model_params,
            "classifier_head_params": classifier_params,
            "embedding_params": embedding_params,
        },
        "blocks": blocks,
        "layers": layers,
        "unused_modules": unused,
    }

    print(f"[model_arch] layers captured: {len(layers)} in forward, {len(unused)} unused")
    print(
        f"[model_arch] embedding params {embedding_params:,} "
        f"(+{classifier_params:,} unused classifier) = {total_model_params:,} total"
    )
    print(
        "[model_arch] stem {} -> {} -> {} -> {} -> GAP {} -> 512 linear -> L2 norm".format(
            captured["conv2d_1a"].shape[1:],
            captured["conv2d_4a"].shape[1:],
            captured["repeat_1"].shape[1:],
            captured["repeat_3"].shape[1:],
            captured["avgpool_1a"].shape[1:],
        )
    )
    for name, info in blocks.items():
        print(
            f"[model_arch] {name}: {info['count']}x {info['block_type']} "
            f"out={info['output_shape'][1:]} params={info['params']:,}"
        )

    out_path = RESULTS_DIR / "model_arch.json"
    save_json(payload, out_path)
    print(f"[model_arch] wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
