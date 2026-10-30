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

**So what:** A threshold below about 0.50 rejects me under lighting conditions I already captured.
That is the convenience side of the security dial, before any stranger is involved.

*Correction:* the previous draft quoted 0.18–0.37 as the noise floor. That range was
between-scan same-condition distance; within-scan jitter was never measured, and it reaches 0.4425.

---

## 2. Brightness is (almost) free

**Observed:** Room light versus phone flash, both from the front (400 frame pairs across the two
repeats): mean 0.2585, range 0.1683–0.3839.

**Means:** That mean sits inside the same-condition between-scan range (0.2268–0.3707) and above
within-scan jitter (0.1864). Making the face much brighter did not move the embedding beyond
ordinary repeat-scan variation.

**So what:** The model is invariant to how much light there is. "Robust to lighting" is true for
brightness. (The previous draft's 0.20–0.38 range was slightly too narrow: the true minimum is
0.1683, though the mean is unchanged.)

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

**Means:** Moving the same light from front to side moved my embedding roughly 0.64–0.71, nearly
four times the frame jitter. The model is not invariant to shadow direction.

**So what:** "Robust to lighting" is only half true. Intensity is free, direction is expensive. The
previous draft said "about three times"; the computed ratio is 3.83×.

---

## 4. Side-lit faces cluster with each other

**Observed:** Flash left versus flash right: 0.5106 mean (0.4182–0.6202), well below front–left
0.7098 and front–right 0.6359.

**Means:** Two half-shadowed faces are closer to each other than either is to a front-lit face,
regardless of which half is dark. The model appears to have a "half-shadowed" region in its space.

**So what:** Enrolling only front-lit photos leaves side-lit unlocks furthest away. If enrolment
includes a side-lit scan, both sides are probably covered. (Confirmed from the old draft.)

---

## 5. Light from below is the worst case

**Observed:** Flash right versus flash bottom is the largest direction pair: 0.8879 mean, up to
0.9671. Flash left versus bottom is next at 0.8079. Bottom lighting is also the least stable frame
to frame: flash/bottom_1 has the widest within-scan jitter (mean 0.3345, max 0.4425).

**Means:** Under-lighting is the condition the model handles least well, and it is also the least
stable. Faces are almost never lit from below in photos, so the training data likely contained the
least of it.

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

**So what:** Enrol with several frames and average them; that is a small, free robustness win. Do
not expect it to fix the worst lighting mismatch.

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

## 8. Cross-domain sanity check: where a stranger lands (not calibrated)

**Observed:** Using the same frozen model on the 400 Olivetti faces: the closest Olivetti embedding
to any of my 12 scan means is 0.8869 (nearest scan: flash_bottom_2); the mean over the 400 Olivetti
faces of their closest-scan distance is 1.2875. At frame level the minimum is 0.8803.

**Means:** A stranger's closest approach (0.8869) is *below* my worst same-person frame distance
(0.9671). On this evidence no threshold separates all of me from all of them. But the comparison is
domain-confounded: different capture device, crop and resolution.

**Caveat (must be repeated wherever this number is used):** *different capture/crop/resolution;
upper bound only.* This is not a calibrated impostor distribution.

**So what:** It kills the hope of picking a threshold from the live data alone. A real threshold
needs at least one stranger captured the same way.

---

## 9. What the Olivetti threshold analysis does and does not tell us

**Observed:** In the controlled 5-enrol/5-probe protocol (`results/threshold.json`), the raw-pixel
map reaches EER 0.08706 and the embedding map reaches EER 0.00224; at a matched 1 % false-reject
rate the raw-pixel map accepts 48.59 % of impostors versus 0.044 % for the embedding map.

**Means:** In a controlled open-set protocol, the representation is the dominant lever on where the
security/convenience dial sits — the embedding buys roughly a 38.8× lower equal-error rate.

**What it does not tell us:** nothing about the *webcam* threshold. That protocol uses pre-aligned
Olivetti stills and no webcam-domain impostors, so it cannot calibrate the live system. It does not
substitute for the missing stranger scan.

**So what:** The live study says direction is expensive and jitter sets a floor; the Olivetti
threshold study says the embedding separates identities far better. Joining them into a real
FaceID threshold is the open item.

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

- **A stranger scan under room light.** Every number above except section 8 is me vs me. Without one
  other person captured the same way there is no way to place a threshold; the cross-domain check is
  confounded and only bounds the problem.
- A second subject and a second capture session would test whether condition metadata repeats.
- The old "mild test" observation (room light versus a dimmer shot with an under-chin shadow) stays
  excluded: it also changed distance, head angle and expression, so its distance cannot be
  attributed to lighting. It is a good reminder to change one variable per experiment, not a number.
- The threshold analysis on Olivetti does not calibrate the webcam domain; a live stranger capture is
  the next real step, not more Olivetti analysis.
