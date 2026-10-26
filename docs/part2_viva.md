# Part 2 viva: likely questions and short answers

Numbers come from [part2_forensics.md](part2_forensics.md) and [part2_experiment_log.md](part2_experiment_log.md).
Sample sizes are tiny; say so whenever you quote one.

## Scope
**Q: What does Part 2 add to the detector?**
Evidence around the verdict:
- where in the face the model's decision came from (localization);
- whether the verdict survives mild everyday changes (robustness, stability, ESS);
- the same for videos (frames, temporal stability, intervals);
- all of it in one JSON for Part 3.
Part 2 never changes the verdict. The detector is the source of truth.

**Q: Why didn't you retrain or improve the detector?**
It is Part 1's deliverable and fixed at tag `part1-detector-v2`. Part 2 measures its evidence; changing it would mix
two questions. Findings that matter for Part 1, such as the blur/JPEG sensitivity, are reported to the Part 1 owner.

## Localization
**Q: What is Grad-CAM and why did you use it?**
- It weights the last feature maps by how much they raise a chosen score, and sums them into a heatmap.
- We use it on the score logit(MANIPULATED) - logit(REAL), at `bn2` (1792 x 12 x 12).
- It needs one forward and one backward pass (under 1 s on CPU) and no extra library.

**Q: You first proposed Grad-CAM++. Why did you switch?**
We measured it.
- On this head (global average pooling + linear), the gradient is the same at every position. Grad-CAM++ then
  reduces to "positive-weight channels only".
- Its MANIPULATED and REAL maps were identical: correlation +1.000 on all 11 photos.
- Grad-CAM was class-sensitive (-0.15 to -0.34). At this layer it equals CAM, and its signed map adds up exactly to
  the logit difference (checked to 4 decimals).

**Q: How do you know the heatmap isn't just an edge detector?**
- Model-randomisation test (Adebayo et al. 2018): we randomised the top layers. The map's correlation with the
  original fell to -0.26 to +0.21.
- Class sensitivity, as above.
- A consistency check: the attribution pass must give exactly the detector's logits.

**Q: Is the highlighted region where the image was manipulated?**
No. It is a *model-indicated suspicious region*: where the model's MANIPULATED evidence came from. We have no
manipulation masks, so localization accuracy is "not yet measured".

**Q: Every heatmap has a bright spot. Isn't that misleading on a REAL image?**
Yes, which is why each analysis gets an evidence-strength label. Below fake-class probability 0.5, regions are
labelled "weak MANIPULATED evidence; shown for transparency only" and drawn faint and gray. On our photos the
positive evidence was never zero, even at probabilities below 0.17.

**Q: How are regions extracted?**
- Threshold the map at 0.5 x its maximum and take connected components.
- Drop components under half a feature cell (0.35% of the crop); report at most 3, ranked by share of the map.
- A 1% minimum was tried first and wrongly dropped the peak region on 4 of 11 photos.

## Robustness
**Q: Which perturbations and why?**
- JPEG 90/70/50, downscale x0.75/x0.5, blur sigma 1/2, brightness and contrast +-10%, noise sigma 2.
- They are mild, everyday changes (re-sharing, resizing, soft focus, lighting, sensor noise) that should not change
  a verdict.
- Levels are in the config. We do not claim they cover real-world distortion.

**Q: When the photo is perturbed, the face crop can move. How do you compare heatmaps fairly?**
Two modes.
- *Pipeline* perturbs the photo and reruns face detection. Maps are compared in original-image pixels, over the
  overlap of the two crops.
- *Fixed-crop* perturbs the model's own 380 px input, so maps align pixel by pixel.
- Median crop IoU was 0.97-0.99, so crop shift was small here.

**Q: What did robustness show?**
On 11 photos (264 runs), 22 predictions flipped and no face was lost.
- 20 of the 22 flips went REAL -> MANIPULATED.
- Blur sigma 2 flipped 5 of 11 photos in pipeline mode.
- Some flips happened inside Part 1's own training augmentation range.
This is a finding about the detector on a small sample, not a general robustness figure.

**Q: Why might blur make a real face look fake to the detector?**
Hypothesis only: the detector may partly use sharpness or compression cues that differed between real and fake
training images. We did not test this. It needs a labelled set with controlled quality.

**Q: Is fixed-crop mode applied before or after the JPEG q95 re-save?**
After, the same order as Part 1's training augmentation: stored q95 crop -> degradation -> normalisation. It uses the
same resampling methods too.

## Stability and Evidence Stability Score
**Q: What is the Evidence Stability Score?**
- A project-defined 0-1 summary of how the evidence held up under the 12 perturbations (pipeline mode):
  ESS = 0.4 x prediction + 0.3 x confidence + 0.3 x localization stability.
- Bands: HIGH >= 0.80, MEDIUM >= 0.60, otherwise LOW.
- It is **not a probability of manipulation**, and the weights and bands are our choices.

**Q: Why those weights?**
- The verdict holding is the main question, so prediction gets 0.4. Confidence and localization share the rest.
- Sensitivity check: under 0.5/0.3/0.2 and under equal weights, 0 of 11 photos changed band. Under 0.45/0.45/0.1,
  1 of 11 did.
- So on this sample the bands barely depend on the exact weights (n = 11).

**Q: How does the score handle a prediction near 0.5?**
- Confidence stability measures margin retention: each change is divided by the distance to 0.5, so the same wobble
  costs more near the boundary, and a flip scores 0.
- Photo 2 (0.65) had confidence stability 0.18, while 0.0004-0.05 baselines kept 0.48-0.93.
- We deliberately don't excuse a borderline flip as "expected": a borderline verdict really is less stable.

**Q: Why the cap rule?**
- A weighted mean can stay HIGH even if a mild re-save changed the verdict (photo 1: 0.85 with one flip).
- Any flip or lost face caps the band at MEDIUM. The score is unchanged, and the cap is recorded.

**Q: Why use pipeline mode for the ESS and not fixed-crop?**
The ESS asks whether the conclusion would survive a re-saved or resized file. Face detection and cropping are part
of that chain. Fixed-crop is reported as a diagnostic, and mixing both modes would count each perturbation twice.

**Q: Show that the ESS adds information.**
Photos 8 and 10 were both confidently REAL (0.045 and 0.0004) and got the same strength label. Photo 8 flipped in 5
of 12 runs and scored 0.57 LOW. Photo 10 never flipped and scored 0.83 HIGH.

## Video
**Q: How do you analyse a video without a temporal model?**
- Sample 2 frames per second and run the image detector plus Grad-CAM on each.
- Average the fake-class probability over frames with a face.
- Measure temporal stability: the share of consecutive pairs with the same label and a change under 0.2.
- Mark suspicious intervals: at least 2 sampled frames at >= 0.5, bridging 1-frame gaps.

**Q: Why the mean and not the maximum or a majority vote?**
- A single frame is noisy (robustness showed this), and the maximum lets one bad frame decide.
- The mean dilutes a partly manipulated video, which is why suspicious intervals are reported separately and never
  averaged away.

**Q: Why not run robustness on every frame?**
- The full set costs about 10 s per frame on CPU.
- We run it on 3 key frames: the two most suspicious and the median. This biases the video ESS downward on purpose
  (conservative), and it's documented.

**Q: What happens when a frame has no face?**
It is a finding, not dropped. No-face frames are reported as time ranges with `usable_face_fraction`. Coverage below
50% adds a warning. A video with no face at all is NO_FACE, with no verdict.

**Q: Your AI clip was flagged and your real clip wasn't. Does the detector work on video?**
- We can't say. It's one clip of each.
- The real phone clip shows compression alone didn't flip that real face.
- But the two clips differ in source quality (phone versus 540p AI), so quality remains a possible confound.
- Video accuracy is not yet measured.

## Structured evidence
**Q: What does Part 3 receive?**
One JSON per analysis (schema 1.0), with the same shape for images and videos; fields that don't apply are null or
[]. It contains:
- prediction, confidence, regions, the three stabilities, ESS, temporal stability, intervals;
- robustness observations, warnings, limitations;
- model version, checkpoint and input sha256, and every setting.
It is validated against a JSON Schema before it is written. Examples are in `docs/part2_evidence_examples/`.

**Q: How do you stop the LLM from overstating?**
- The document carries its own wording rules (`score_semantics`) and the limitations.
- NO_FACE means no verdict.
- The schema doc tells Part 3 to explain only fields present in the JSON, to say "fake-class probability of 0.65",
  and to always show limitations.

**Q: How is it reproducible?**
- Seed 42 for noise; everything else is deterministic on CPU.
- The config sha256 and every parameter are stored in each document.
- Robustness reports are saved, and the ESS can be recomputed from them without the model.
- There are 77 Part 2 tests.

## Limitations (have these ready)
- Localization accuracy: not yet measured (no masks). Video accuracy: not yet measured.
- The detector is not robust to mild blur and JPEG on our sample, and doesn't generalise to every generator (StyleGAN
  at chance).
- The ESS and all thresholds are design choices, not calibrated.
- All Part 2 measurements use 11 photos and 2 clips.
