# Results

Machine-readable measurements from the analysis runners. JSON is the reference when checking
claims in the README, findings and paper. **The paper's LaTeX numbers are currently manually
transcribed**, not imported from JSON; compare them after every regeneration.

| file | produced by | what it contains |
|---|---|---|
| `olivetti.json` | `analysis/run_olivetti.py` | Map A (4096-d raw pixels) vs Map B (512-d embeddings): silhouette, 1-NN LOO, genuine/impostor pairs, ROC-AUC, EER, TAR@FAR, d′, variance milestones, 2-D rank correlation, seed robustness, full-rank PCA isometry sanity check (not a causal dimensionality control), confusion analysis |
| `live.json` | `analysis/run_live.py` | webcam illumination study: within-scan jitter, same-condition repeats, intensity vs direction effects, pooling, worst genuine distance, frame-lag/outlier analysis, cross-domain check, 12×12 distance matrix |
| `threshold.json` | `analysis/run_threshold.py` | known-population Olivetti verification: original in-sample per-split metrics and operating points (not deployment calibration), plus identity-disjoint calibration/evaluation and unknown-to-gallery probes; interpolated EER vs empirical FAR/FRR are distinguished |
| `model_arch.json` | `analysis/model_arch.py` | verified InceptionResnetV1 layer names/shapes/parameter counts from forward hooks, checkpoint hash |

Every file has a `common` header with `schema_version`, `created`, `git_commit`, Python/package
versions, device, seed and notes. The full schema is in `docs/DESIGN.md` §6; every value is a plain
JSON number (no NumPy scalars).

## Regenerate

```sh
make all           # from an activated environment; tests + all four runners
```

With the same inputs, checkpoint and software stack, reruns are seeded and expected to reproduce
measurements; `created` and `git_commit` change, and cross-device numerical differences can occur.
`results/*.npz` are ignored caches (embeddings, map coordinates) regenerated when stale.
