# Data

Everything here is either tracked raw data or a regenerable cache.

## Tracked

| path | what it is |
|---|---|
| `scans/` | 12 webcam scans, one per lighting condition/repeat. Each is a `(10, 512)` float32 matrix of L2-normalised InceptionResnetV1 embeddings. ~24 KB each. |
| `scans.json` | The manifest. Makes the raw `.npy` files self-describing; `fem.data.load_scans()` reads it and refuses unknown/inconsistent entries. |
| `README.md` | This file. |

### Manifest schema

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

- `intensity`: `room` (ambient) or `flash` (phone flash).
- `direction`: `front` | `left` | `right` | `bottom`.
- `repeat`: 1 or 2 (the second scan of the same condition, used as the repeat baseline).
- `file` is relative to `data/scans/`.

The design is unbalanced by necessity: room light was only available from the front and left, so
there are 12 scans rather than 16. `fem.data.load_scans()` reports this in the JSON results.

### Adding a scan

```sh
$PY scripts/live_capture.py --intensity flash --direction left --repeat 3 --frames 10
```

The capture tool writes `<name>.npy` plus a `.json` sidecar. **Add the matching entry to
`scans.json` by hand** — updating the manifest automatically is on the roadmap
(`docs/ROADMAP.md` §7). Then `make live` picks it up.

## Ignored caches (regenerable, never committed)

| path | what it is | regenerate with |
|---|---|---|
| `sklearn/` | Olivetti faces downloaded once by `fetch_olivetti_faces(data_home=...)` (~4 MB) | `make maps` |
| `../.cache/mpl/` | matplotlib cache (the home directory cache is not writable) | any `make` target |
| `../results/*.npz` | deterministic caches: 400 Olivetti embeddings, 2-D map coordinates, threshold curves | `make maps` |
