# Face Embedding Map — Project Plan

**The question this project answers:**
Do learned embeddings separate *identity* better than raw pixels do — and can I show it with a number, not just a nice-looking picture?

**Rule for myself:** no training happens until Stage 7. Stages 1–6 use a frozen, pretrained model. Nothing learns, nothing fits, there is no train/test split.

---

## Stage 0 — Setup

Environment with: numpy, matplotlib, scikit-learn, torch, facenet-pytorch.

Reuse the Docker/Jupyter setup from the MNIST CNN if that's easier than a fresh venv.

**Done when:** `fetch_olivetti_faces()` runs without error.

---

## Stage 1 — Look at the data

*Time: ~20 minutes. Do this before anything else.*

Olivetti ships inside scikit-learn — `sklearn.datasets.fetch_olivetti_faces`. Downloads once, caches after.

What comes back:
- `.images` — 400 faces, each a 64×64 grid
- `.data` — the same 400 faces, each already flattened to 4096 numbers
- `.target` — 400 numbers, 0–39, saying who each face is

40 people, exactly 10 photos each.

**Do:**
1. Print the shape of `.images` and of `.data`. Confirm 4096 = 64 × 64.
2. Use matplotlib's `imshow` to display one face. (`imshow` = draw an array as a picture. Use `cmap='gray'`.)
3. Display all 10 photos of person 0 side by side.
4. Print the first 20 numbers of one flattened face.

**Find and write down:**
- What actually varies between the 10 photos of one person? Lighting? Angle? Glasses? Expression?
- What's the range of the pixel values — 0 to 1, or 0 to 255?
- Are the faces already aligned (eyes in roughly the same spot every time)?

That last one matters more than it looks. Olivetti is cropped tight and mostly aligned, which quietly makes Stage 2 easier than it deserves to be. Note it now so I remember later why LFW is harder.

---

## Stage 2 — Map A: raw pixels

*This is Eigenfaces, the 1991 method. It's a real method, not a strawman.*

**Do:**
1. Take `.data` — already flattened, 400 points in 4096-dimensional space.
2. Run PCA down to 2 dimensions.
3. Scatter plot the 400 points. Colour each dot by `.target` so each person gets a colour.

**Then the interesting bit — look at the recipe:**
4. PCA's components have 4096 numbers each, same as an image. Reshape the first few back to 64×64 and `imshow` them. These are eigenfaces.
5. Also plot the explained variance ratio.

**Find and write down:**
- Do same-coloured dots sit together, or are they scattered?
- What does the first eigenface look like? What is PC1 actually measuring?
- How much of the variance do the first 2 components hold? What am I throwing away by squashing to 2D?

**Expected result:** clusters exist, but they track lighting and head angle more than identity. If Olivetti looks better than expected, that's the alignment doing the work — worth noting rather than explaining away.

---

## Stage 3 — Map B: learned embeddings

**Do:**
1. Load `InceptionResnetV1(pretrained='vggface2')` from facenet-pytorch. Put it in eval mode. Nothing trains.
2. Feed the faces through it. Out come 512 numbers per face.
3. Run **the same PCA, same settings** as Stage 2 on those 512-dim vectors, down to 2D.
4. Same scatter plot, same colours.

**Watch out for:**
- The model expects 160×160 RGB images. Olivetti is 64×64 greyscale. Something has to convert — decide what and note the decision, because resizing 64→160 is inventing detail that isn't there.
- Normalisation has to match what the model was trained with.

**Find and write down:**
- Are the clusters tighter than Map A?
- Are the *same* people confused in both maps, or different ones?

---

## Stage 4 — Measure it, don't eyeball it

Two pictures side by side is an opinion. This stage turns it into a result.

**Do:** pick a cluster-quality metric that uses the true labels — silhouette score against `.target` is the obvious one, and it's already in scikit-learn. Compute it for Map A and Map B.

**Critical:** compute it on the **full-dimensional** vectors (4096 and 512), not the 2D ones. The 2D version is for human eyes only. Measuring on it would be measuring my plotting settings.

**Find and write down:** two numbers, and which is bigger.

**Also:** re-run everything with a different PCA random seed. Do the numbers move? If they swing a lot, the result isn't solid yet.

---

## Stage 5 — Me as a test case (the FaceID bit)

This is the part I actually wanted to build. Nothing above needed my face; this does.

**Do:**
1. Take 5–10 photos of myself. Different lighting on purpose — one bright, one dim, one lamp-to-the-side.
2. Use MTCNN (comes with facenet-pytorch) to detect and crop the face. Don't crop by hand — MTCNN aligns on the eyes, which is the thing that stops the two-pixel-shift problem.
3. Push each through the model. Save the vectors. This is the "enrolment" step — the password.
4. Take a *new* photo, not one of the enrolled ones. Get its vector.
5. Measure distance from the new vector to each saved one. Cosine distance is standard for these.

**Find and write down:**
- Distance from my new photo to my saved photos: ______
- Distance from my new photo to a random Olivetti person: ______
- Is there a clear gap between those two numbers?

**Then the real question:** where do I put the threshold? Below X = "it's Nic", above X = "unknown". Try a few values and see what breaks. Set it too tight and my own dim photo gets rejected. Too loose and a stranger gets in.

That trade-off is the entire security-vs-convenience dial in every real face unlock system. Feeling it by hand is the point of this stage.

**Optional:** repeat step 5 with Map A's PCA recipe instead of the model, and compare. If Eigenfaces lets a stranger in where the CNN doesn't, that's the headline result of the whole project.

---

## Stage 6 — Yale B, the lighting experiment (optional but strong)

Extended Yale Face Database B: ~38 people, ~64 photos each, same pose, lighting deliberately swept around the room. The dataset is *built* to isolate the variable I care about.

**Do:** rerun Stages 2–4 on it.

**Expected:** Map A clusters by lamp position instead of by person — the lighting problem made visible. Map B shouldn't care.

**Sanity check:** the 2016 paper *A Comprehensive Analysis of Deep Learning Based Representation for Face Recognition*, section 4.3, runs this exact comparison. Read it after getting my own numbers, not before.

---

## Stage 7 — Train my own (separate project, honestly)

Only after everything above works. This is where "train" finally means something.

The idea: instead of learning labels, feed the network three faces at a time — an anchor, another photo of the same person, and a photo of someone else. Push the first two together, shove the third away. That's triplet loss, and it's how the pretrained model learned to ignore lighting in the first place.

Read: Schroff et al., *FaceNet* (arXiv 1503.03832) — makes far more sense after Stages 1–6.

Balanced classes matter *here*, not earlier.

---

## Writing it up

The repo README should open with the question and close with the two numbers from Stage 4. Both maps as images. A short "what I got wrong" section — the t-SNE-as-embedder confusion is worth including, plenty of people have it.

A repo with a claim beats a repo with a demo.

---

## Traps, collected

1. **t-SNE is a plotter, not an embedder.** No reusable recipe, different picture every run. Fine for looking, useless for matching. PCA has a recipe; use PCA anywhere the output needs to be reusable.
2. **Identical squashing settings for both maps**, or the comparison is measuring my dial-twiddling.
3. **Measure in full dimensions**, plot in 2D. Never measure on the plot.
4. **Don't build the pipeline twice.** Load → embed → squash → plot → score. Only the embed step swaps. If I'm copy-pasting plotting code, stop and pull it into a function.
5. **No training before Stage 7.** If I've written a training loop early, I've drifted.
