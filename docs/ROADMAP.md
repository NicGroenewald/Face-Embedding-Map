# Next best steps

*Prioritized after the October 2026 rebuild. The project has a working comparison and a paper;
what it does not yet have is a consented, identity- and session-disjoint impostor distribution for
the live pipeline. Without one, the webcam study measures same-person variability only; with one, a
held-out calibration becomes possible (still small-sample, not a deployment guarantee).*

Legend: **P0** = blocks the main claim / next paper; **P1** = strengthens the evidence a reviewer
will ask for; **P2** = engineering and dissemination; **P3** = research extensions.

---

## P0 — make the threshold a measurement, not a description

### 1. Capture strangers under the same six conditions

The live study currently has one subject and no impostors (`findings/live-illumination.md` says so).
Every "worst genuine" number (0.967) is me-vs-me; there is no number that says what a stranger
scores, so no threshold can be chosen from this data.

**Do:** first pilot a consented capture protocol with 3–5 additional people under comparable
conditions, then recruit enough identities and independent sessions to reserve disjoint people for
calibration and evaluation. Save new templates privately (`scripts/live_capture.py` defaults to
`data/private-scans/`); do not append them to the fixed twelve-scan historical manifest.
**Then:** analyse genuine, known-population impostor and unknown-to-gallery trials *within the same
webcam domain*. Select thresholds on calibration identities and report empirical FAR/FRR on held-out
identities/sessions; avoid interpreting a few volunteers as evidence for very small FAR.
**Acceptance:** a new subject-aware manifest/runner and reproducible live verification results,
with consent/privacy controls, capture design and confidence intervals by identity. Only then can
the paper discuss a webcam operating point. Capture and ethics cannot be automated by this repo.

### 2. Recalibrate the threshold per condition

The observed direction contrast (0.714, 3.83× frame jitter) motivates—but does not establish—the
hypothesis that one global threshold may be insufficient. On held-out subjects, test: (a) a single
global threshold, (b) a threshold conditioned on a cheap
lighting-quality proxy (e.g. face-region brightness asymmetry), (c) multi-frame aggregation at
verification.
**Acceptance:** a table of EER/FAR/FRR for each rule, and one sentence saying which rule a
practitioner should use.

### 3. Repeat across sessions and sensors

The current data is one session, one laptop webcam. Template ageing and sensor change are the
failure modes a real system cares about. Capture the same subjects on a second day and (if
available) a phone camera; report the genuine-distance shift.
**Acceptance:** a "same person, different day/sensor" distance row alongside the existing
"same person, different light" row.

---

## P1 — strengthen the controlled comparison

### 4. Extended Yale Face Database B (the stage the plan predicted)

Yale B sweeps illumination around ~38 subjects at fixed pose. It is the dataset that can isolate
lighting *direction* with many identities, which is exactly the variable the webcam study found
expensive. Re-run E1–E4 on it with the same `fem` pipeline.
**Why it matters:** the paper currently generalises a one-person lighting result; Yale B replaces
that with an identity-controlled claim.
**Watch out:** check the license/access route before starting; the pipeline already takes any
`(N, H, W)` image array, so the code change is small.

### 5. LFW for the unconstrained benchmark

Olivetti is the best case for raw pixels; LFW is the honest one. Same frozen model, same metrics,
plus a proper identity-disjoint verification protocol. This also tests whether the 38.8× gap is a
property of the model or of the easy dataset.

### 6. Two cheap ablations reviewers will ask for

- **Alignment ablation:** run Map A on MTCNN-cropped Olivetti (instead of pre-aligned) and Map B on
  both crops. This separates "alignment does the work" from "representation does the work".
- **Synthetic relighting:** apply smooth directional gradients to Olivetti images, re-embed, and
  measure the identity-preserving shift. It isolates direction from capture artefacts and connects
  the webcam observations to the illumination-cone literature.

### 7. Improve the live protocol code

- `scripts/live_capture.py` saves private scans and sidecars but intentionally does **not** mutate
  the historical `data/scans.json`; design a separate consent-aware, subject-aware manifest and
  runner before importing new scans. Keep raw templates out of public git by default.
- Add a per-frame quality score (face-box probability, sharpness, pose) and store it, then test
  quality-weighted frame pooling at enrolment (the paper shows plain pooling buys only 4.8 %).
- Add a small `--list` mode that prints the current manifest and per-condition distances.

---

## P2 — engineering and dissemination

### 8. CI

A single GitHub Actions workflow: `pytest`, `make all` on a runner with cached weights (or skip the
model-dependent jobs and keep tests + paper build), and `make paper` to catch LaTeX breakage. The
repo is currently verified manually.

### 9. License and citation

Choose a license deliberately (MIT/Apache-2.0 for code, CC-BY-4.0 for the paper are the usual
split) and add it. `CITATION.cff` is already present; update it when the paper is deposited.

### 10. Archive the paper

Put `paper/main.pdf` on arXiv (cs.CV) or Zenodo and add the DOI to the README and `CITATION.cff`.
The original machine reproduced results from a fresh clone; independent packaging, external data
consent and scientific claim review still precede an archival release.

### 11. Lock the environment

`requirements.txt` pins exact versions; add `uv lock` (or a `requirements.lock` with hashes) so
"reproducible" does not depend on wheel availability.

---

## P3 — research extensions (only after P0/P1)

### 12. Train something (Stage 7)

Triplet loss on Olivetti, with held-out identities, to see how much of the gap is available from
400 images without a large pretrained corpus. This is the only stage where "training" is
meaningful, and it should keep the same verification protocol so numbers stay comparable.

### 13. Compare modern embedders

ArcFace/AdaFace and a current FRVT-evaluated system on the same protocol. This tests whether the
findings are properties of deep embeddings generally or of the 2018 VGGFace2 checkpoint.

### 14. Failure analysis and fairness

Per-identity error analysis on a harder dataset, and demographic/sensor diversity. The current paper
explicitly cannot speak to fairness or template ageing; a follow-up should.

---

## Do not do (traps this project already paid for)

1. Do not measure on the 2-D plot. The rank-correlation number (0.807 vs 0.517) exists to make this
   impossible to forget.
2. Do not compare distances across domains (Olivetti ↔ webcam) as if they were an impostor
   distribution; the cross-domain caveat is part of the result.
3. Do not tune a threshold on the same trials used to report its error rates — split subjects, not
   just images.
4. Do not add t-SNE as an "embedding"; it is a plotter with no reusable transform.
5. Do not train before the data protocol can measure what improved.

---

## Suggested next three actions

1. Draft the stranger-capture consent/one-page protocol and book 3 volunteers (P0.1).
2. While that runs, implement the Yale B loader as a drop-in `fem.data.load_images` path (P1.4).
3. Land P0.1 + P0.2 as a single follow-up paper section; keep the current paper as the controlled
   Olivetti result.
