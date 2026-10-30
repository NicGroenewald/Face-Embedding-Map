# Data and biometric privacy

The twelve historical scans are committed **face-embedding templates** of one named subject. They
are not raw photos, but embeddings can support matching/re-identification and must be treated as
sensitive biometric data. The capture date and subject are also recorded in `scans.json`. Check
consent, licensing and repository visibility **before publishing or pushing** this data. A credential
scanner cannot assess biometric privacy; no tracked scan has been removed or rewritten here.

## Tracked historical study

| path | what it is |
|---|---|
| `scans/` | Twelve `(10, 512)` float32 matrices of L2-normalised InceptionResnetV1 vectors. |
| `scans.json` | Curated manifest giving each historical scan its condition and repeat. |

`fem.data.load_scans()` validates shape, finite values and unit norms. `analysis/run_live.py` is a
**fixed twelve-scan, two-repeats-per-condition experiment**; it does not automatically analyse extra
scans. Room light has front/left only, phone flash front/left/right/bottom.

```json
{
  "format_version": 1,
  "subject": "nic",
  "captured": "2026-09-06",
  "frames_per_scan": 10,
  "scans": [
    {"name": "room_front_1", "file": "06092026-light_face_front_1.npy",
     "intensity": "room", "direction": "front", "repeat": 1}
  ]
}
```

A `file` entry is relative to `data/scans/`, not to the manifest's location. The historical
condition/repeat names end in `_1` or `_2`; `Scan.repeat` currently derives its value from that
suffix. Avoid treating a sidecar's timestamped capture ID as a study condition label.

## Capturing new data safely

```sh
python scripts/live_capture.py --intensity flash --direction left --repeat 3 --frames 10
```

New captures default to **gitignored `data/private-scans/`** and receive a `.npy` template plus a
`.json` sidecar with a unique capture ID (`name`) and the requested human `label`. The command
never modifies the published manifest. Keep new scans private; obtain informed consent and agree on
a study protocol before deliberately moving a copy into `scans/` or changing `scans.json`.

**Do not simply append a third scan to the manifest and run `make live`:** the historical runner
asserts twelve scans and exactly two repeats per condition. A new multi-subject/live-verification
analysis needs an explicit new runner, a subject-aware manifest, identity-disjoint calibration/test
splits, and a fresh paper/results schema. That work remains on `docs/ROADMAP.md`.

## Ignored caches

| path | what it is | regenerate with |
|---|---|---|
| `sklearn/` | downloaded Olivetti faces (~4 MB) | `make maps` |
| `../.cache/mpl/` | matplotlib cache | any plotting target |
| `../results/*.npz` | Olivetti embeddings and analysis caches | `make maps`; `make all` for all caches |
