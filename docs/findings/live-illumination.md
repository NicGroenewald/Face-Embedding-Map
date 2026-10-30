# Live Embeddings (Stage 5): Findings

*Rewritten against `results/live.json` (12 webcam scans, 6 Sep 2026, one subject). Every number
below is a computed value from that file; the previous draft's 0.94 worst case is superseded by
0.9671. Pipeline in `scripts/live_capture.py`, analysis in `analysis/run_live.py`.*

Pipeline: webcam frame → MTCNN detect and crop (CPU) → InceptionResnetV1 pretrained on VGGFace2
(MPS) → 512-number unit vector. Ten frames per scan, saved as a `(10, 512)` matrix. Distance is
Euclidean between unit vectors, so 0 is identical and 2 is opposite; cosine similarity is
`1 - d²/2`.

Conditions: room light (front, left) and phone flash (front, left, right, bottom), two scans each.
All scans are me. One subject, no stranger.

---

## 0. Getting it running at all

**Observed:** The tutorial script crashed on the first frame with an MPS adaptive-pool error. A
fresh script returned no face on the first frame with no error, and the first frames came back
near-black while the camera warmed up.

**Means:** Two Mac-specific problems, neither of which any tutorial mentions. MTCNN cannot run on
Apple's GPU, so it goes on CPU while the embedder stays on MPS. The webcam's first frames are dark
while it warms up.

**So what:** Each tutorial was fine code for someone else's machine. Detector on CPU, discard the
first frames, and everything downstream works. Section 7 shows why discarding them is right, not
just convenient.

---

## 1. Frame jitter: the noise floor

**Observed:** Within one scan there are 45 frame pairs. Per-scan mean distances run from 0.1144 to
0.3345, with an overall mean of 0.1864; the worst single within-scan pair is 0.4425 and the best is
0.0406. Between the two scans of the *same* condition (100 frame pairs each): room/front 0.2637
mean (0.1875–0.3744), room/left 0.2268 (0.1833–0.3247), flash/front 0.2464 (0.1844–0.3553),
flash/left 0.3095 (0.2282–0.4971), flash/right 0.2628 (0.1760–0.3393), flash/bottom 0.3707
(0.2496–0.4695).

**Means:** Even with nothing changed, same-person distances run about 0.19 within a scan and about
0.23–0.37 between scans. The worst same-condition pair reaches 0.4971.

**So what:** A one-reference verifier using the particular frame pair at distance 0.4971
would reject that probe below 0.4971. A five-reference verifier might find a closer match;
these pairs alone do not set a usable security threshold.

*Correction:* the previous draft quoted 0.18–0.37 as the noise floor. That range was
between-scan same-condition distance; within-scan jitter was never measured, and it reaches 0.4425.

---

## 2. Brightness is (almost) free

**Observed:** Room light versus phone flash, both from the front (400 frame pairs across the two
repeats): mean 0.2585, range 0.1683–0.3839.

**Means:** The observed contrast sits inside the same-condition between-scan range
(0.2268–0.3707) and above within-scan jitter (0.1864). Separate scans do not isolate
brightness from pose, sensor auto-exposure or time.

**So what:** These captures do not show a large additional front-lit room-versus-flash effect;
this is **not** evidence of general brightness invariance. The previous draft's 0.20–0.38 range
was slightly too narrow: the observed minimum is 0.1683.

---

## 3. Light direction is expensive

**Observed:** All six flash direction pairs, mean (min–max) of frame-pair distances:

| directions | mean | min | max |
|---|---|---|---|
| front – left | 0.7098 | 0.6195 | 0.7996 |
| front – right | 0.6359 | 0.5734 | 0.6930 |
| front – bottom | 0.7335 | 0.6040 | 0.8157 |
| left – right | 0.5106 | 0.4182 | 0.6202 |
| left – bottom | 0.8079 | 0.7363 | 0.8745 |
| right – bottom | 0.8879 | 0.7948 | 0.9671 |

Mean direction effect: 0.7143. That is **3.83×** the mean within-scan jitter of 0.1864.

**Means:** Across these sequential captures, differently labelled flash directions accompany
mean distances of roughly 0.64–0.71 from front to side, substantially above frame jitter.

**So what:** Direction is a strong candidate explanation, but pose and auto-exposure were not
controlled. The descriptive direction/jitter ratio is 3.83×, not proof of a causal effect.

---

## 4. Side-lit faces cluster with each other

**Observed:** Flash left versus flash right: 0.5106 mean (0.4182–0.6202), well below front–left
0.7098 and front–right 0.6359.

**Means:** Two half-shadowed faces are closer to each other than either is to a front-lit face,
regardless of which half is dark. The model appears to have a "half-shadowed" region in its space.

**So what:** Enrolling multiple conditions, including a side-lit scan, is a testable hypothesis;
these pair distances do not measure FAR/FRR for that enrolment policy.

---

## 5. Light from below is the worst case

**Observed:** Flash right versus flash bottom is the largest direction pair: 0.8879 mean, up to
0.9671. Flash left versus bottom is next at 0.8079. Bottom lighting is also the least stable frame
to frame: flash/bottom_1 has the widest within-scan jitter (mean 0.3345, max 0.4425).

**Means:** Under-lighting accompanies the largest variation and cross-condition distances in
this small sample. A training-image explanation is plausible but was not tested.

**So what:** 0.9671 is the worst me-vs-me frame distance found, on a 0–2 Euclidean scale; cosine
similarity 0.5324. The previous draft's 0.94 understated the true maximum.

---

## 6. The worst genuine distance, and what averaging buys

**Observed:** The worst frame pair is flash_right_2 versus flash_bottom_2 at 0.9671 (cosine
similarity 0.5324). Averaging the 10 frames of each scan (renormalised scan means) reduces that
same pair to 0.9149 (cosine 0.5815); the worst pooled pair over all 12 scans is also 0.9149.
Across all 66 scan pairs, the frame-level mean distance is 0.6042 versus 0.5751 at scan-mean level,
a 4.8 % reduction. Scan-mean vectors had norm 0.9906 before renormalisation (range
0.9728–0.9967), so pooling barely changes direction; renormalise anyway to keep the cosine relation.

**Means:** Averaging frames helps, but only modestly for genuine pairs — because the frames within a
scan are not independent (section 7). It does not rescue a side-lit/bottom-lit pair.

**So what:** Averaging reduces observed mean distances modestly. It is not yet shown to improve
verification error rates; compare enrolment rules on held-out identities before recommending one.

---

## 7. Frames within a scan are not independent

**Observed:** Mean frame-pair distance by lag: lag 1 0.1534, lag 2 0.1639, lag 3 0.1856, lag 4
0.1905, lag 5 0.1997, lag 6 0.2128, lag 7 0.2245, lag 8 0.2364, lag 9 0.2568 (pairs available:
108 down to 12). Frame 0's mean distance to the other nine frames is 0.2434; every other frame is
0.1671–0.1938.

**Means:** Distance grows steadily with frame lag — slow head/pose drift during a scan — and frame 0
is the clear outlier, consistent with camera warm-up.

**So what:** Ten frames are not ten independent samples; the effective sample size is below 10, so
"n = 10 frames" overstates the evidence. Discard the first frames (the refactored `scripts/live_capture.py` does),
and do not treat within-scan maxima as independent draws.

---

## 8. Cross-domain sanity check (not a webcam stranger sample)

**Observed:** Using the same frozen model on the 400 Olivetti faces: the closest Olivetti embedding
to any of my 12 scan means is 0.8869 (nearest scan: flash_bottom_2); the mean over the 400 Olivetti
faces of their closest-scan distance is 1.2875. At frame level the minimum is 0.8803.

**Means:** The minimum across these *different-domain* subjects (0.8869) is below the maximum
same-person webcam frame-pair distance (0.9671), but the scores arise from different capture,
crop and resolution. The pair maximum is not a bound for a five-reference verifier either.

**Caveat (must be repeated wherever this number is used):** *This is neither an upper nor a lower
bound on same-camera strangers, and cannot calibrate an operating threshold.*

**So what:** Same-domain impostor captures and identity/session-disjoint evaluation are needed.

---

## 9. What the Olivetti threshold analysis does and does not tell us

**Observed:** In the controlled 5-enrol/5-probe protocol (`results/threshold.json`), the raw-pixel
map reaches EER 0.08706 and the embedding map reaches EER 0.00224; at a matched 1 % false-reject
rate the raw-pixel map accepts 48.59 % of impostors versus 0.044 % for the embedding map.

**Means:** On the same Olivetti trials used to select the original threshold, the embedding has a
38.8× lower *interpolated* EER. The 0.22 % figure is near/below the 0.5 % genuine-trial grid; it
is not a directly achievable FAR=FRR threshold.

**What it does not tell us:** This known-population comparison reuses identities and, for the
original numbers, uses the same trials for calibration and evaluation. It does not calibrate a
webcam, nor test novel webcam strangers. See the separate identity-disjoint calibration/evaluation
block in `results/threshold.json` for a more honest Olivetti-only check.

**So what:** These data suggest the embedding separates Olivetti identities better; a calibrated
live verifier still needs consented same-camera subjects and independent test sessions.

---

## 10. What changed from the previous draft

- Worst genuine distance: 0.94 → **0.9671** (flash_right_2 vs flash_bottom_2).
- "About three times frame jitter" → **3.83×**.
- The 0.18–0.37 "noise floor" was between-scan same-condition distance, not within-scan jitter;
  within-scan jitter is now measured (0.1864 mean, 0.4425 max).
- The old "room front vs flash bottom 0.63–0.79" claim is dropped: it changed intensity *and*
  direction at once. Only matched comparisons are reported now.
- New here: within-scan jitter, the direction/noise ratio, the pooling effect, temporal
  autocorrelation, the first-frame outlier, and the cross-domain check.

---

## 11. Still to do for Stage 5

- **Consented same-camera strangers and held-out people/sessions.** Every within-live distance above
  is me vs me; cross-domain scores do not bound the missing webcam impostor distribution. One extra
  person is a pilot, not adequate evidence for a very low FAR.
- A second capture session and a larger subject sample should test whether condition effects repeat.
- The old "mild test" observation (room light versus a dimmer shot with an under-chin shadow) stays
  excluded: it also changed distance, head angle and expression, so its distance cannot be
  attributed to lighting. It is a good reminder to change one variable per experiment, not a number.
- The threshold analysis on Olivetti does not calibrate the webcam domain; a live stranger capture is
  the next real step, not more Olivetti analysis.
