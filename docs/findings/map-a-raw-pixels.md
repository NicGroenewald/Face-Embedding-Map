# Map A — Raw Pixels: Findings

*Rewritten against `results/olivetti.json` (E1; run 30 Oct 2026). Every number below is a computed
value from that file. The 2-D scatter is for eyes only; all metrics are full-dimensional. Pipeline in
`analysis/run_olivetti.py`, library in `fem/`.*

Stage 2 of the plan. Eigenfaces on Olivetti: 400 photos, 40 people, 10 each, 64×64 greyscale, PCA
straight on the 4096 raw pixel values. No training, no model.

---

## 1. The data is pre-aligned, and that quietly helps

**Observed:** The dataset-wide pixel range is `[0.0, 1.0]` during the whole run, not a narrow band.
Across the 10 photos of person 0 the eyes sit in nearly the same place every time. What moves is
mouth corners, eyebrow height and head tilt.

**Means:** Olivetti is tightly cropped and roughly eye-aligned. The two-pixel-shift problem that
breaks raw-pixel comparison is mostly absent here. (The previous draft said pixels run 0.08–0.88;
that is the range of image 0 only. The dataset as a whole spans the full `[0, 1]` scale.)

**So what:** Any result Map A gets on Olivetti is a best case for raw pixels. LFW and a live webcam
will not be this kind. Note it now so a good-looking Map A is not over-read.

---

## 2. The first components are lighting, not people

**Observed:** Reshaped to 64×64, component 0 is a broad brightness gradient across the whole face;
component 1 is bright on the left and dark on the right (qualitative reading of
`figures/eigenfaces`). Putting a number on that reading: the seed-0 2-D coordinates correlate with
per-image pixel statistics — PC1 with overall brightness at Pearson r = 0.9860 (Spearman 0.9790)
and PC2 with the left-minus-right brightness gradient at r = 0.9445
(`map_a.pca_axes_interpretation`). The two visible lobes in the scatter are therefore dark versus
bright photographs, not groups of people. Component 0 alone carries 23.81 % of the variance
(`component0_evr` 0.23813). The cumulative milestones are 2 components for 25 %, 4 for 50 %, 19 for
75 %, 66 for 90 %; 95 % is not reached within the first 100 components (recorded as `null`).

**Means:** A quarter of everything that differs between 400 photos of 40 people is which direction
the light came from. PCA finds the biggest sources of pixel change, and on faces the biggest source
is illumination.

**So what:** Raw pixels spend their strongest directions on how the photo was lit. This is the
baseline Map B has to beat, and the exact failure this project set out to show. The old numbers
(23.8 %, 19 components, 66 components) are confirmed.

---

## 3. Eyeballing components underestimated them, twice

**Observed:** My first pass over the component images guessed that only a handful mattered; reading
the scree plot I revised that upward. Both guesses were low. The measured numbers are the milestone
table in section 2: 19 components for 75 %, 66 for 90 %.

**Means:** The flat tail of a scree plot is not empty. Many small components add up. Visual
inspection of a component image says nothing about how much variance it explains.

**So what:** Measure, do not eyeball. There is also no single "right" number of components — it
depends on the threshold chosen, which is why the guesses had nothing to land on.

---

## 4. Decision: no StandardScaler before PCA

**Observed:** With per-pixel scaling the corner pixels, which are near-constant background, got
amplified into visible noise in the component images.

**Means:** All 4096 columns share one unit (brightness 0–1), so per-column scaling only inflates
low-variance background. Classic eigenfaces centres the data but does not scale it.

**So what:** PCA is fit on centred raw pixels only. This is also the setting for Map B, so the
comparison changes the representation and nothing else.

**Rejected:** StandardScaler, for the reason above.

---

## 5. The 2-D scatter is a plot, not a measurement

**Observed:** The 2-D PCA scatter of all 400 points is now in `figures/mapA_scatter` (and in
`notebooks/01-map-a-raw-pixels.ipynb`). Its Spearman correlation with the full-dimensional pairwise distances is 0.8070. For
comparison, Map B's 2-D scatter preserves less: 0.5174. Re-running the projection with PCA seeds
0, 1 and 2 gives 0.8070 every time, and the rank correlation between the three 2-D geometries is
1.0 — the picture does not move.

**Means:** Map A's 2-D picture actually preserves the true distance ordering *better* than Map B's,
even though Map B separates identities far better (section 6). Eyeballing the two scatters would
rank the maps wrongly.

**So what:** Never measure on the plot (plan trap #3). And seed robustness is not a question for the
full-dimensional metrics: silhouette, k-NN and ROC are deterministic, so they are seed-independent;
only the randomized-SVD projection could move, and it does not.

---

## 6. Full-dimensional metrics (Map A)

**Observed:**
- silhouette (40 labels, 4096-d): **0.10557**
- 1-NN leave-one-out accuracy: **0.9450**; top-5: **0.9825**
- genuine pairs: n = 1800, mean **8.3777**, std 2.5559, min 1.9091, max 22.642
- impostor pairs: n = 78000, mean **12.4143**, std 2.4498, min 5.6968, max 24.8271
- ROC-AUC: **0.88071**; EER **0.2050** at distance threshold 10.388
- TAR at FAR = 1e-3: **0.26389**; TAR at FAR = 1e-2: **0.42333**
- d′: **1.6124**

**Means:** Raw pixels cluster weakly and identify surprisingly well — 94.5 % of faces find a
same-person nearest neighbour — but verification fails badly: a fifth of genuine/impostor decisions
are wrong at the equal-error point. Identification and verification are different questions, and
this map is good at one and bad at the other. That disagreement is a finding, not a contradiction.

**So what:** This is the number Map B gets compared against. Under the identical protocol (same
images, same PCA settings, same metrics), Map B scores silhouette 0.49698, 1-NN 1.0, ROC-AUC
0.99954, EER 0.011111 and d′ 4.988 — the full comparison is in the paper and
`results/olivetti.json`.

---

## 7. Which people does Map A fail on?

**Observed:** Per-identity 1-NN leave-one-out accuracy ranges from 0.6 to 1.0 across the 40
identities. 378 of the 400 images get the same predicted identity under Map A and Map B (0.945 of
images). Map B classifies all 400 correctly, so per-identity accuracy has no spread there: the
Pearson correlation between the two maps' per-identity accuracies is undefined (`null` in the JSON),
and no image is misclassified by both maps, so "both wrong, same wrong choice" has no denominator
(`both_wrong_count` = 0, fraction `null`).

**Means:** Map A's errors are concentrated in a subset of identities; the embedding fixes all of
them on this sample. The interesting "are the same people confused by both maps?" question cannot be
answered positively when one map makes no errors at all.

**So what:** To test that question, the experiment needs a harder dataset (LFW, unaligned webcam
frames), not more Olivetti. On this data the answer is "Map B has no confusions to compare".

---

## 8. Full-rank PCA isometry check, not a dimensionality ablation

**Observed:** 400 centred samples span at most rank 399, so a 512-component PCA of the raw pixels
cannot exist. Projecting onto the 399-component full-rank basis retains all observed variance and
changes pairwise distances by at most 3.47e-12. Unsurprisingly every distance-based metric is
identical to Map A's (e.g. silhouette 0.10557364).

**Means:** This checks that an orthogonal change of basis did not alter Euclidean distances. It
is **not** a matched-dimensionality control that isolates feature count from pretrained
representation and training-corpus effects; both observed 400-sample point clouds have rank at
most 399 after centring regardless of nominal feature width.

**So what:** The embedding wins on these data, but this isometry check alone cannot attribute
causality to representation rather than dimensionality. The check remains in
`results/olivetti.json` under its original key `controls.map_a_pca512` for compatibility.

---

## 9. Limitations

- **Olivetti is favourable to raw pixels:** images are pre-aligned and tightly cropped; it is
  reasonable to expect additional unconstrained variation to make raw-pixel matching harder, but
  that expectation is not a proven mathematical upper bound.
- **No local model training:** the CNN is frozen but pretrained on a much larger corpus. The
  original 5/5 verification operating points reuse the same Olivetti identities/trials for
  threshold selection and measurement.
- **PCA for the 2-D plot is fit on all 400 images, including probes.** Headline silhouette,
  identification and verification metrics use the unprojected full vectors, so this transductive
  plot fit does not demonstrate that the measured gap is conservative.
- **The 2-D scatter must not be measured.** It preserves only part of the true geometry (0.8070 for
  Map A, 0.5174 for Map B).
- **95 % variance is not reached in the first 100 components** (`null` in the JSON), so statements
  about "how many components Map A needs" stop at 90 %.

---

## 10. What is now done from the old "still to do" list

- The 2-D scatter of all 400 points coloured by person exists (`figures/mapA_scatter`,
  `notebooks/01-map-a-raw-pixels.ipynb`).
- Silhouette, 1-NN, ROC/EER and the genuine/impostor comparison are computed in full dimensionality
  (section 6), not on the 2-D plot.
- The plotting code is a shared `fem.plotting` function, so Map B reuses identical settings
  (plan traps #2 and #4).
- The unused `X_scaled` / StandardScaler code is removed from the notebook.
