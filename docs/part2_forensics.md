# Part 2: localization, robustness, video and structured evidence

Part 2 takes the Part 1 detector as it is (EfficientNet-B4 v2 "mixed", checkpoint sha256 `d00ae780...abd107`) and
adds forensic evidence around its prediction. The output is one JSON document per analysis for Part 3. The detector
is the source of truth: Part 2 measures how its evidence behaves and never changes the verdict.

```
image / video
  -> Part 1 detector (face crop 380 px, EfficientNet-B4)        prediction, fake-class probability
  -> localization (Grad-CAM on bn2, MANIPULATED - REAL)         model-indicated suspicious regions + strength label
  -> robustness (12 controlled perturbations)                   flips, probability changes, heatmap changes
  -> stability (prediction / confidence / localization, ESS)    0-1 scores, HIGH / MEDIUM / LOW (our bands)
  -> video (2 fps frames, mean, temporal stability, intervals)  robustness + ESS on 3 key frames
  -> structured evidence JSON (schema 1.0)                      docs/part2_evidence_schema.md
```

## Contents
1. [How to run](#how-to-run)
2. [Components and design decisions](#components-and-design-decisions)
3. [Results at a glance](#results-at-a-glance)
4. [Known limitations and open questions](#known-limitations-and-open-questions)
5. [Reproducibility](#reproducibility)

Related: [evidence schema](part2_evidence_schema.md), [experiment log](part2_experiment_log.md),
[viva Q&A](part2_viva.md), [detector API (Part 1)](detector_api.md).

## How to run
All commands are for Git Bash on the laptop CPU, from the repository root, using the venv. Quote paths that contain
spaces. Outputs must go to a folder outside the repository; the scripts refuse folders inside it. The default is
`~/Downloads/deeptrace_outputs/<step>`.

| purpose | command | CPU time |
|---|---|---|
| full evidence JSON (image or video) | `.venv/Scripts/python.exe scripts/analyze.py --input "<file>"` | ~13 s per image, ~45 s for a 5 s video |
| heatmap and regions only | `.venv/Scripts/python.exe scripts/localize_image.py --image "<file>" ...` | ~1 s per image |
| robustness, both modes, with summary table | `.venv/Scripts/python.exe scripts/robustness_image.py --image "<file>" ...` | ~19 s per image |
| stability and ESS from saved robustness reports | `.venv/Scripts/python.exe scripts/stability_report.py --robustness-dir "<dir>"` | instant |
| video report and key-frame overlays | `.venv/Scripts/python.exe scripts/analyze_video.py --video "<file>"` | ~45 s for a 5 s clip |
| tests (the real-detector checks need the checkpoint) | `DEEPTRACE_TEST_FACE="<face photo>" .venv/Scripts/python.exe -m pytest -q` | ~30 s |

Every setting lives in [configs/part2_forensics.yaml](../configs/part2_forensics.yaml). Each value is commented as
a design choice, and the evidence JSON records the config's sha256.

## Components and design decisions
Code lives in `src/deeptrace/forensics/`. Part 1 files are unchanged.

### 1. Localization (`attribution.py`, `localization.py`, `overlay.py`)
- **Method:** Grad-CAM at `bn2` (the 1792 x 12 x 12 feature map that is pooled into the classifier), on the
  logit difference MANIPULATED - REAL.
  - With a global-average-pooling + linear head, Grad-CAM at this layer equals CAM, and its signed map decomposes
    the logit difference exactly (verified to 4 decimals).
  - Grad-CAM++ was the first choice. On this head it gave the same map for both classes (correlation +1.000), so it
    is kept only as a config option.
- **Safety checks:**
  - The attribution pass must reproduce the detector's logits (tolerance 1e-4), or the analysis fails.
  - The hook is removed after each call, and no parameter gradients are accumulated.
  - A model-randomisation sanity test is included.
- **Regions:**
  - Threshold at 0.5 x the map maximum, then take connected components.
  - The minimum area is half a feature cell (0.35% of the crop); at most 3 regions are reported.
  - Boxes are given in crop and original pixels.
- **Evidence-strength label:** every normalised map has a bright area, so each analysis is labelled
  `model_indicated` (fake-class probability >= 0.5), `weak` (< 0.5: regions shown for transparency only), `none`
  or `not_applicable`. Weak regions are drawn faint and gray, with an amber banner.
- **Wording:** "model-indicated suspicious regions", never "manipulated pixels". Probabilities are never printed as
  0.00 or 1.00 ("below 0.01" / "above 0.99").

### 2. Robustness (`robustness.py`)
- **Perturbations (12):**
  - JPEG quality 90 / 70 / 50.
  - Downscale x0.75 / x0.5 and back.
  - Gaussian blur sigma 1 / 2.
  - Brightness and contrast x0.9 / x1.1.
  - Gaussian noise sigma 2 (seeded).
- **Two modes:**
  - *pipeline* perturbs the original image and reruns face detection, crop and detector (realistic). Maps are
    compared in original-image pixels, over the overlap of the two crops.
  - *fixed_crop* perturbs the baseline crop after the detector's q95 re-save. This is the same order as Part 1's
    training augmentation. Maps are compared pixel by pixel. Comparing the two modes separates detector sensitivity
    from crop shift.
- **Findings, not errors:** a prediction flip or a lost face (NO_FACE) is recorded as an observation.
- **Sanity check:** an identity perturbation reproduced the baseline exactly on all 11 photos in both modes.

### 3. Stability and Evidence Stability Score (`stability.py`)
- **prediction stability:** the share of runs that kept the label. A lost face counts as not kept.
- **confidence stability (margin retention):** the mean of `max(0, 1 - |p - p0| / max(|p0 - 0.5|, 0.01))`. The same
  change costs more near the 0.5 boundary, and a flip always scores 0.
- **localization stability:** the mean of max(0, map correlation). A run without a comparable map scores 0.
- **ESS** = 0.4 x prediction + 0.3 x confidence + 0.3 x localization, from **pipeline mode only**. Fixed-crop values
  are reported as a diagnostic.
  - Bands (our choice): HIGH >= 0.80, MEDIUM >= 0.60, LOW otherwise.
  - **Cap rule:** any flip or NO_FACE caps the band at MEDIUM. The score is unchanged, and `band_capped` and
    `cap_reason` are recorded.
  - The output reports the per-component cost and the driver (the component that cost the most points).
- The ESS is project-defined and **not a probability of manipulation**.

### 4. Video (`video.py`)
- **Sampling and per-frame analysis:** frames are sampled at 2 fps (at most 120). Each frame gets the image detector
  and Grad-CAM (about 0.8 s per frame on CPU).
- **Aggregation:** the mean fake-class probability over frames with a usable face. Median, maximum and the fraction
  of MANIPULATED frames are also reported.
- **Temporal stability:** the share of consecutive face-frame pairs with the same label and a change under 0.2. Flips,
  standard deviation and face jumps (crop IoU < 0.3) are reported too.
- **Suspicious intervals:** runs of at least 2 sampled frames at >= 0.5, with 1-frame gaps bridged. Shorter runs are
  reported as isolated frames.
- **No-face frames** are findings, grouped into time ranges, plus `usable_face_fraction`. Coverage below 50% adds a
  warning.
- **Robustness and ESS on 3 key frames only:** the two most suspicious frames plus the median frame, in pipeline mode.
  The full set on every frame would take about 10 s per frame. Choosing the most suspicious frames biases the video
  ESS downward; this is deliberate (conservative) and documented in the output.

### 5. Structured evidence (`evidence.py`)
- **One shape for images and videos**, versioned `schema_version` "1.0". Fields that do not apply are `null` or `[]`.
- **Contents:**
  - Prediction, confidence and probabilities.
  - Suspicious regions and the three stability components.
  - ESS, temporal stability, suspicious intervals.
  - Robustness observations, video observations, warnings, limitations.
  - Model version and checkpoint sha256, input sha256, and the config sha256 with every setting.
- **NO_FACE** propagates with no verdict and no scores.
- **Small faces:** a face crop under 122 px native adds a Part 2 warning and limitation. Part 1's own warning uses
  96 px; the stricter 122 px follows Part 1's measured accuracy drop.
- **Validation:** every document is checked against `docs/part2_evidence_schema.json` before it is written.

## Results at a glance
All results below are measured, on very small samples whose true labels are unknown, except where stated. They
describe how this detector behaved on these inputs. **None of them is an accuracy claim.** Details are in the
[experiment log](part2_experiment_log.md).

### Attribution check (n = 11 photos)
| check | Grad-CAM++ | Grad-CAM (chosen) |
|---|---|---|
| MANIPULATED map vs REAL map correlation | +1.000 on all 11 | -0.15 to -0.34 |
| signed map + bias = logit difference | no | yes, to 4 decimals |
| trained vs randomised-weights map correlation | about 0 | -0.26 to +0.21 |

### Robustness (n = 11 photos x 12 perturbations x 2 modes = 264 runs; 1 photo predicted MANIPULATED, 10 REAL)
| mode | perturbation | flips / 11 | median abs. change in fake-class probability | median map correlation |
|---|---|---|---|---|
| pipeline | JPEG 90 / 70 / 50 | 1 / 2 / 2 | 0.06 / 0.13 / 0.01 | 0.75 / 0.70 / 0.62 |
| pipeline | downscale x0.75 / x0.5 | 0 / 2 | 0.01 / 0.12 | 0.66 / 0.49 |
| pipeline | blur sigma 1 / 2 | 2 / 5 | 0.26 / 0.45 | 0.47 / 0.30 |
| pipeline | brightness x0.9 / x1.1 | 0 / 1 | 0.01 / 0.00 | 0.89 / 0.80 |
| pipeline | contrast x0.9 / x1.1 | 0 / 0 | 0.01 / 0.01 | 0.85 / 0.87 |
| pipeline | noise sigma 2 | 0 | 0.02 | 0.73 |
| fixed_crop | JPEG 90 / 70 / 50 | 1 / 0 / 0 | 0.01 / 0.05 / 0.04 | 0.82 / 0.78 / 0.68 |
| fixed_crop | downscale x0.75 / x0.5 | 0 / 1 | 0.00 / 0.01 | 0.90 / 0.76 |
| fixed_crop | blur sigma 1 / 2 | 1 / 2 | 0.01 / 0.24 | 0.82 / 0.52 |
| fixed_crop | brightness, contrast (4 levels) | 0 | <= 0.01 | 0.91-0.97 |
| fixed_crop | noise sigma 2 | 2 | 0.20 | 0.67 |

- 22 flips and 0 NO_FACE in 264 runs. 20 of the 22 flips went REAL -> MANIPULATED.
- 6 of the 10 REAL-predicted photos flipped at least once. Median crop IoU in pipeline mode was 0.97-0.99.

### Evidence Stability Score (n = 11 photos, pipeline mode, 12 runs each)
| photo | prediction | fake-class p | flips | prediction | confidence | localization | ESS | band | driver |
|---|---|---|---|---|---|---|---|---|---|
| 1 | REAL | 0.022 | 1 | 0.92 | 0.80 | 0.82 | 0.85 | MEDIUM (capped from HIGH) | confidence |
| 2 | MANIPULATED | 0.652 | 2 | 0.83 | 0.18 | 0.84 | 0.64 | MEDIUM | confidence |
| 3 | REAL | 0.002 | 0 | 1.00 | 0.84 | 0.65 | 0.85 | HIGH | localization |
| 4 | REAL | 0.002 | 1 | 0.92 | 0.69 | 0.63 | 0.76 | MEDIUM | localization |
| 5 | REAL | 0.001 | 0 | 1.00 | 0.86 | 0.70 | 0.87 | HIGH | localization |
| 6 | REAL | 0.165 | 2 | 0.83 | 0.56 | 0.63 | 0.69 | MEDIUM | confidence |
| 7 | REAL | 0.001 | 2 | 0.83 | 0.77 | 0.57 | 0.74 | MEDIUM | localization |
| 8 | REAL | 0.045 | 5 | 0.58 | 0.48 | 0.62 | 0.57 | LOW | prediction |
| 9 | REAL | 0.009 | 0 | 1.00 | 0.93 | 0.72 | 0.90 | HIGH | localization |
| 10 | REAL | 0.0004 | 0 | 1.00 | 0.88 | 0.56 | 0.83 | HIGH | localization |
| 11 | REAL | 0.096 | 2 | 0.83 | 0.61 | 0.59 | 0.69 | MEDIUM | localization |

Photos 8 and 10 share the Step 2 label `weak`, but the ESS separates them: 0.57 LOW (5 flips) versus 0.83 HIGH
(0 flips).

### Weight sensitivity of the ESS bands (n = 11 photos)
| weights (prediction / confidence / localization) | photos changing band |
|---|---|
| 0.5 / 0.3 / 0.2 | 0 of 11 |
| 0.34 / 0.33 / 0.33 | 0 of 11 |
| 0.45 / 0.45 / 0.1 | 1 of 11 (photo 2: MEDIUM -> LOW, 0.64 -> 0.54) |

### Video control (n = 1 AI-generated clip, n = 1 real phone clip; true sources known)
| clip | sampled frames with a face | per-frame fake-class probability | video result | temporal stability | intervals | video ESS |
|---|---|---|---|---|---|---|
| AI-generated (PixVerse), 5 s, 1024x576 | 11 / 11 | 0.96-0.9994 | MANIPULATED, mean 0.99 | 1.00 | 0.0-5.0 s | 0.97 HIGH |
| real phone selfie (consented), 9 s | 19 / 19 | 0.0009-0.21 | REAL, mean 0.03 | 1.00 | none | 0.86 HIGH |

These clips test the pipeline. One clip per class says nothing about video detection accuracy.

## Known limitations and open questions
### Detector (inherited from Part 1; Part 2 does not change it)
- Confidence is an uncalibrated softmax probability. It is not forensic certainty and not the probability that the
  media is manipulated.
- The detector does not generalise to every generator: ROC-AUC 0.9966 on DeepFakeFace test, 0.82 on held-out
  OpenFake generators, and 0.49 (chance) on StyleGAN faces.
- Only the largest face is analysed. Images and frames without a detectable face get no prediction.
- Faces under about 122 px native crop were less reliable. Part 2 warns below 122 px; Part 1's own warning uses 96 px.

### Localization
- Regions are model-indicated: they explain the model, not the manipulation. **Localization accuracy is not yet
  measured**, because no manipulation masks exist for our data (DeepFakeFace has none).
- Grad-CAM at bn2 is coarse: a 12 x 12 grid, about 32 px per cell of the 380 px crop.
- Maps are normalised per image, so a bright area always exists. The strength label and `positive_mass` mitigate
  this; they do not remove it.
- The region threshold (0.5) and minimum area (0.35%) were chosen by inspecting 11 photos, not tuned on masks.
- Grad-CAM++ was not class-sensitive on this head. We did not test other layers or methods (occlusion, Score-CAM).

### Robustness and stability
- 12 mild perturbations at 1-3 levels do not cover real-world distortion: no rotation, cropping, colour shifts,
  screenshots, social-media pipelines or adversarial edits.
- Results come from 11 photos with unknown true labels; only 1 was predicted MANIPULATED.
- **Blur and JPEG sensitivity:** mild blur and recompression flipped REAL predictions to MANIPULATED (20 of 22 flips
  went that way). Some flips happened inside Part 1's own training augmentation range (blur sigma <= 1.5, resize
  >= 0.5, JPEG >= q40).
  - *Open question (hypothesis):* does the detector rely partly on sharpness or compression cues rather than
    manipulation traces? Testing this needs a labelled set with controlled quality, which is Part 1's area.
- Pipeline mode hits harder than fixed-crop mode.
  - *Hypothesis:* the perturbation acts at native face size (140-420 px), before upscaling to 380 px. Not measured
    separately.
- The ESS weights, bands, cap rule, margin floor and the 0.2 jump threshold are our design choices and are not
  calibrated. The ESS is not a probability. The weight sensitivity shown above was measured on 11 photos only.
- Confidence stability is symmetric: a swing away from the boundary (0.65 -> 0.95) also counts as instability.
- Region IoU is reported but not used in the score, because it is too sensitive to the threshold.

### Video
- Frames are analysed independently; there is no temporal model. Only sampled frames (2 fps) are seen, so events
  shorter than about 0.5 s can be missed.
- Only the largest face per frame is analysed. In multi-person videos the analysed face can change (reported as face
  jumps).
- The key-frame ESS is biased downward by design (the most suspicious frames).
- **Resolution / source-quality confound:** the AI clip (540p-class, MANIPULATED) and the real phone clip (REAL)
  differ in source quality and encoding. The phone clip shows that compression alone did not flip that real face, but
  source quality is not ruled out. One clip per class is not evidence of accuracy.
  - *Open check (not done):* re-encode the real clip at 540p and a low bitrate, and rerun.
- No no-face frames occurred in either clip, so the no-face path is tested only on synthetic video.
- Video accuracy is not yet measured.

### Engineering
- CPU runtime: about 13 s per image for full evidence, and about 45-51 s for a 5-9 s clip. The model is loaded once
  per process.
- The built-in schema validator covers a subset of JSON Schema. Part 3 can validate with any full validator against
  the same file.

## Reproducibility
- Seed 42 (`run.seed`) drives the noise perturbation. Everything else is deterministic on CPU.
- Each evidence document stores the config file sha256, every setting used, the model version and checkpoint sha256,
  and the input sha256.
- Robustness and stability outputs store their settings. `stability_report.py` recomputes the ESS from saved
  robustness reports without the model.
- Tests: 77 Part 2 tests under `tests/test_forensics_*.py`. Three real-detector checks run only when the checkpoint
  and `DEEPTRACE_TEST_FACE` are present.
