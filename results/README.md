# Results

Machine-readable outputs of the analysis runners. These files are the **source of truth** for every
number in the README, the findings documents and `paper/main.pdf`; nothing is hand-typed.

| file | produced by | what it contains |
|---|---|---|
| `olivetti.json` | `analysis/run_olivetti.py` | Map A (4096-d raw pixels) vs Map B (512-d embeddings): silhouette, 1-NN LOO, genuine/impostor pairs, ROC-AUC, EER, TAR@FAR, d′, variance milestones, 2-D rank correlation, seed robustness, matched-dimensionality control, confusion analysis |
| `live.json` | `analysis/run_live.py` | webcam illumination study: within-scan jitter, same-condition repeats, intensity vs direction effects, pooling, worst genuine distance, frame-lag/outlier analysis, cross-domain check, 12×12 distance matrix |
| `threshold.json` | `analysis/run_threshold.py` | open-set verification: per-split FAR/FRR, EER, TAR@FAR, AUC, matched-FRR operating points, score-rule sensitivity, session-bias check |
| `model_arch.json` | `analysis/model_arch.py` | verified InceptionResnetV1 layer names/shapes/parameter counts from forward hooks, checkpoint hash |

Every file has a `common` header with `schema_version`, `created`, `git_commit`, Python/package
versions, device, seed and notes. The full schema is in `docs/DESIGN.md` §6; every value is a plain
JSON number (no NumPy scalars).

## Regenerate

```sh
make all           # tests + all four runners (~25 s on Apple Silicon with cached weights)
```

The runners are deterministic: a rerun reproduces every value byte-for-byte apart from the
`created` and `git_commit` header fields. `results/*.npz` are ignored deterministic caches
(embeddings, map coordinates) — regenerated on demand, never committed.
