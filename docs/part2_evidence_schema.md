# DeepTrace evidence JSON (Part 2 -> Part 3), schema version 1.0

Part 2 writes **one JSON document per analysed image or video**. Part 3 (LLM, FastAPI, Next.js) should depend only
on this document, not on Part 2's Python code.

- Machine-readable schema: [part2_evidence_schema.json](part2_evidence_schema.json) (JSON Schema draft 2020-12; any
  validator works).
- Examples: [image](part2_evidence_examples/example_image.json), [video](part2_evidence_examples/example_video.json),
  [no face](part2_evidence_examples/example_no_face.json). These are real pipeline outputs with the file name, input
  sha256 (all zeros), timestamp and artifact names replaced. The numbers are examples, not accuracy results.
- Produce a document: `python scripts/analyze.py --input "<image or video>" --out-dir "<folder outside the repo>"`.
  The script checks every document against the schema before writing it. On CPU this takes about 13 s per image and
  about 45 s for a 5 s video.

## Rules for Part 3
1. **The detector is the source of truth.** The LLM explains these fields. It must not add a verdict, a score or a
   region that is not in the document.
2. **`status == "NO_FACE"` means there is no verdict.** `prediction`, `confidence` and every stability field are
   `null`. Say that no face was found. Do not show REAL or MANIPULATED.
3. **Scores are not certainty.** Write "The detector assigned a fake-class probability of 0.65", never "65% chance it
   is fake". `score_semantics` contains this wording rule in text, so you can pass it to the LLM.
4. **Regions are model-indicated suspicious regions**, never "manipulated pixels". Always show
   `localization.message`. When `localization.evidence_strength == "weak"`, the regions are shown for transparency
   only, because the detector leaned REAL.
5. **The Evidence Stability Score is not a probability.** It is a project-defined 0-1 score with our own
   HIGH / MEDIUM / LOW bands. Show `evidence_stability.summary`, and show `cap_reason` when `band_capped` is true.
6. **Always show `limitations` and `warnings`.** Always show `robustness_observations` (prediction flips under mild
   perturbations) and, for videos, `observations` (frames without a face, face jumps).
7. Treat unknown extra fields as optional. A breaking change will change `schema_version`.

## Top-level fields
Every key is always present. Fields that do not apply are `null` or `[]`.

| field | type | image | video | meaning |
|---|---|---|---|---|
| `schema_version` | string | "1.0" | "1.0" | contract version |
| `media_type` | "image" / "video" | yes | yes | |
| `generated_at` | string (UTC, ISO 8601) | yes | yes | |
| `status` | "OK" / "NO_FACE" | yes | yes | NO_FACE: no prediction exists (video: no sampled frame had a face) |
| `input` | object | file_name, sha256, size_bytes, width, height | + fps, frame_count, duration_s | the analysed file |
| `model` | object | yes | yes | detector name, `version` (v2), `checkpoint_sha256`, epoch, training data |
| `score_semantics` | string | yes | yes | how to read and phrase the probabilities |
| `prediction` | "REAL" / "MANIPULATED" / null | yes | mean of the frames >= 0.5 | |
| `confidence` | number / null | probability of the predicted class | max(mean, 1 - mean) | uncalibrated |
| `fake_probability`, `real_probability` | number / null | softmax probabilities | mean over frames with a face | |
| `face` | object | found, face_box, face_score, crop_box, native_crop_side | usable_face_frames, usable_face_fraction, sampled_frames | |
| `aggregation` | object / null | null | mean, median, max, min, manipulated_frame_fraction, summary | |
| `frames` | array / null | null | one entry per sampled frame: frame, time_s, status, prediction, fake_probability, evidence_strength | timeline chart |
| `localization` | object / null | evidence_strength, message, positive_fraction, method, target, target_layer | method, note | |
| `suspicious_regions` | array | up to 3 regions | regions of the key frames (+ frame, time_s) | see below |
| `prediction_stability` | object / null | value plus runs, flips, no_face | median over key frames (+ per_key_frame) | |
| `confidence_stability` | object / null | value plus margin details | median over key frames | |
| `localization_stability` | object / null | value plus compared_runs | median over key frames | |
| `evidence_stability` | object / null | score, band, band_uncapped, band_capped, cap_reason, components, points_lost, driver, summary, weights, bands, mode, note | the same + key_frames, method, bias_note | see below |
| `temporal_stability` | object / null | **null** | value, pairs, steady_pairs, label_flips, label_agreement, fake_probability_std, face_jumps | |
| `suspicious_intervals` | array | [] | start_s, end_s, suspicious_frames, mean/max fake_probability, peak_time_s | |
| `isolated_suspicious_frames` | array | [] | suspicious runs shorter than 2 sampled frames | |
| `robustness` | object / null | modes, runs, results (one per perturbation) | modes, key_frames | |
| `robustness_observations` | array | prediction flips / lost faces under perturbations | the same, tagged with frame and time_s | findings, not errors |
| `observations` | array | [] | no_face_frames, face_jump | findings, not errors |
| `warnings` | array of {code, message} | e.g. small_face, small_face_part2, no_face_detected | e.g. low_face_coverage, small_face | |
| `limitations` | array of string | always | always | show them |
| `settings` | object | yes | yes | config file sha256, seed and every parameter used (reproducibility) |
| `artifacts` | object | overlay file name | key_frame_overlays {frame: file name} | files written next to the JSON |

## Field semantics
- **Suspicious region:** `bbox_original` is [left, top, right, bottom] in original-image pixels (for videos, in the
  frame). `bbox_crop` is the same box in the 380 px face crop. `area_fraction` is the box's pixel area as a fraction
  of the crop. `share_of_map` is the fraction of the heatmap inside the region. `peak` is the maximum of the heatmap,
  which is normalised to a maximum of 1, so a bright area always exists. Read `evidence_strength` before you
  highlight anything.
- **evidence_strength:** `model_indicated` (fake-class probability >= 0.5), `weak` (< 0.5: shown for transparency
  only), `none` (nothing pushed towards MANIPULATED), `not_applicable` (no face).
- **Stability components (all 0-1, higher is more stable):**
  - prediction: the share of perturbations that kept the label (a lost face counts as not kept).
  - confidence: the share of the distance to the 0.5 boundary that was kept.
  - localization: the mean clipped correlation of the heatmap with the original heatmap.
- **Evidence Stability Score** = 0.4 x prediction + 0.3 x confidence + 0.3 x localization. Bands: HIGH >= 0.80,
  MEDIUM >= 0.60, otherwise LOW. Any prediction flip or lost face caps the band at MEDIUM (`band_capped`). `driver`
  is the component that cost the most points. For videos the score is the median over key frames (the two most
  suspicious frames plus the median frame), which is deliberately conservative.
- **Temporal stability:** the share of consecutive face-frame pairs with the same label and a change in fake-class
  probability under 0.2.
- **Suspicious interval:** at least 2 sampled frames (2 per second) with fake-class probability >= 0.5. Gaps of 1
  sampled frame are bridged. Times are the timestamps of the first and last sampled frames.

## Measured context (do not present it as better than this)
On DeepFakeFace test data the detector reached ROC-AUC 0.9966. On held-out OpenFake generators it reached 0.82. On
StyleGAN faces it was at chance (0.49). Mild blur and recompression flipped predictions on a small sample of photos
(docs/part2_experiment_log.md). Video accuracy and localization accuracy are not yet measured.
