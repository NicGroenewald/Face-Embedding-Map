# Figures

All result figures are **generated**, never hand-edited. Each one is written twice by the same call:
`<name>.png` (200 dpi, for README/issues) and `<name>.pdf` (vector, for the paper). Regenerate with
`make all` or the specific runner.

| figure | produced by | what it shows |
|---|---|---|
| `mapA_scatter` | `analysis/run_olivetti.py` | 2-D PCA of raw Olivetti pixels, coloured by identity |
| `mapB_scatter` | `analysis/run_olivetti.py` | same plot, same settings, frozen-embedding space |
| `eigenfaces` | `analysis/run_olivetti.py` | first 16 eigenfaces (64×64) |
| `scree` | `analysis/run_olivetti.py` | explained variance / cumulative, Map A vs Map B |
| `distance_histograms` | `analysis/run_olivetti.py` | genuine vs impostor pair distances, both maps |
| `roc_curves` | `analysis/run_olivetti.py` | pair-protocol ROC with EER marked |
| `metric_bars` | `analysis/run_olivetti.py` | silhouette / 1-NN / ROC-AUC / EER side by side |
| `confusion_panel` | `analysis/run_olivetti.py` | per-identity 1-NN accuracy, Map A vs Map B |
| `live_heatmap` | `analysis/run_live.py` | 12×12 mean distance between webcam scans |
| `live_lighting` | `analysis/run_live.py` | frame jitter vs intensity vs direction effect |
| `live_pooling` | `analysis/run_live.py` | frame-level vs scan-mean distance distributions |
| `threshold_curves` | `analysis/run_threshold.py` | FAR and FRR vs threshold, both maps |

The paper's **architecture diagrams** are not in this folder: they are TikZ sources in
`paper/figures/src/` compiled to `paper/figures/fig_*.pdf` (run `make diagrams`).

Style rules: colourblind-safe palette, Map A and Map B share colours and axis conventions when
compared, fonts readable at paper column width, grayscale-legible, title states what is measured.
