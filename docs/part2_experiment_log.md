# Experiment log (Part 2: forensics)

Measured observations only. Detector: v2 "mixed", checkpoint sha256 d00ae780e23a7f713d3f7af8e57e05a4a5e08382c005772499e9a7fcedabd107.
Settings: configs/part2_forensics.yaml at the commit that added each entry; seed 42. Test inputs are kept outside the
repository and are not identified here. Nothing below is an accuracy claim: the samples are tiny and their true labels
are not known (except where stated).

## Attribution method choice on 11 still face photos (measured, 2026-10-26)
- Target layer bn2 (conv_head + BatchNorm + SiLU, 1792 x 12 x 12), then global average pooling + linear head. The
  gradient there equals w / 144 at every position (checked numerically).
- Grad-CAM++ on the MANIPULATED logit: the correlation between its MANIPULATED and REAL maps was +1.000 on all 11
  photos (not class-sensitive); raw maximum about 3,600 on every photo.
- Grad-CAM: MANIPULATED vs REAL map correlation -0.15 to -0.34. Its signed map on the logit difference plus the bias
  difference reproduced the detector's logit difference to 4 decimals on all 11 photos.
- Model-randomisation check (last block, conv_head, bn2 and classifier randomised, seed 0): correlation of the trained
  map with the randomised map -0.26 to +0.21.
- Consistency: the attribution pass and the detector gave identical logits (max difference 0) on all 11 photos.
- Decision: Grad-CAM on logit[MANIPULATED] - logit[REAL]; Grad-CAM++ kept as a config option.

## Region extraction on the same photos (measured, 2026-10-26)
- Thresholding at 0.5 x the map maximum: peak regions covered 0.49-0.92% of the crop, bilinear specks at most 0.19%.
  A 1% minimum area wrongly dropped the peak region on 4 of 11 photos, so the minimum was set to 0.35% (half a
  12 x 12 feature cell).
- Even with fake-class probability below 0.17, positive_mass was 1.8-5.7, i.e. every normalised map has a bright
  area. This motivated the weak-evidence label.

## Robustness on 11 still face photos (measured, 2026-10-26)
- 12 perturbations x 2 modes per photo (264 runs). Baseline: 1 photo predicted MANIPULATED (fake-class probability
  0.65), 10 predicted REAL (0.0004 to 0.16).
- 22 prediction flips, 0 NO_FACE. 20 of the 22 flips went REAL -> MANIPULATED; 6 of the 10 REAL-predicted photos
  flipped at least once.
- Strongest effect: Gaussian blur sigma 2 in pipeline mode flipped 5 of 11 photos (median |change| 0.45).
- Some flips happened inside Part 1's training augmentation range (blur sigma <= 1.5, resize >= 0.5, JPEG >= q40),
  e.g. JPEG q90 in fixed-crop mode on one photo (0.05 -> 0.72).
- Median crop IoU under pipeline perturbations was 0.97-0.99, so face-crop shift was small.
- Interpretation (hypothesis, not measured): the detector may partly rely on sharpness or compression cues.

## Evidence Stability Score on the same 11 photos (measured, 2026-10-26)
- ESS (pipeline mode, weights 0.4 / 0.3 / 0.3) ranged 0.57 (LOW) to 0.90 (HIGH); one photo was capped from HIGH to
  MEDIUM by the flip rule.
- Two photos with the same Step 2 label (weak MANIPULATED evidence) separated clearly: 0.57 LOW (5 flips in 12 runs)
  versus 0.83 HIGH (0 flips).
- Weight sensitivity: 0 of 11 bands changed under 0.5 / 0.3 / 0.2 and under equal weights; 1 of 11 changed under
  0.45 / 0.45 / 0.1.

## Video: one AI-generated clip versus one real phone clip (measured, 2026-10-26)
Pipeline tests with n = 1 per class. They show how the detector behaved on these two clips, not video accuracy.

| clip | source | sampled frames (2 fps) | face frames | per-frame fake-class probability | video result | temporal stability | suspicious intervals | video ESS |
|---|---|---|---|---|---|---|---|---|
| AI-generated (PixVerse), 5 s | 1024x576, 24 fps; face crop 306-333 px native | 11 | 11 (100%) | 0.96 to 0.9994 | MANIPULATED, mean 0.99 | 1.00 (10/10 steady) | 0.0-5.0 s | 0.97 HIGH |
| real phone selfie (consented, the project member's own), 9 s | phone camera; resolution not recorded | 19 | 19 (100%) | 0.0009 to 0.21 | REAL, mean 0.03 | 1.00 (18/18 steady) | none | 0.86 HIGH |

- Question: could video compression alone explain the PixVerse MANIPULATED verdict, given that blur and
  recompression pushed REAL photos towards MANIPULATED on still images?
- Observation: the real phone clip is also compressed video, and it stayed REAL on all 19 frames (highest frame 0.21).
  So on this clip, video compression by itself did not push a real face to MANIPULATED.
- Not ruled out: the two clips differ in source and quality (phone camera versus 540p-class AI output, different
  encoders and bitrates). Source quality therefore remains a possible confound. The PixVerse face crops were not small
  (306-333 px native, about 1.2x upscaling to 380 px), so small-face upscaling is an unlikely explanation for that clip.
- In the PixVerse frames the generator watermark lies outside the face crop, so the detector did not see it.
- A video limitation was added to the evidence output (deeptrace.forensics.video.VIDEO_LIMITATIONS).
- To narrow the confound (not yet done): re-encode the real phone clip at 540p and a lower bitrate and rerun.
  Expected only as a test of the confound, not as an accuracy measurement.
