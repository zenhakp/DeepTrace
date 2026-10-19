# Detector API (Part 1 -> Part 2 / Part 3)

Model: EfficientNet-B4 (ImageNet initialisation), fine-tuned on DeepFakeFace face crops. Run "main", checkpoint epoch 3.
Labels: index 0 = REAL, index 1 = MANIPULATED.

## Use
```python
from deeptrace.models.inference import DeepTraceDetector
det = DeepTraceDetector.from_run("main")      # needs <DEEPTRACE_ROOT>/checkpoints/main/best.pt
result = det.analyze("photo.jpg")             # JSON-serialisable dict
```
Command line: `python scripts/predict.py --image photo.jpg [--save-crop crop.jpg]`

## Setup
1. Install requirements.txt plus torch/torchvision; `pip install -e .`
2. Set DEEPTRACE_ROOT (default ./deeptrace_data).
3. Put best.pt (68 MB, from the team's private Drive; not in git) at `<DEEPTRACE_ROOT>/checkpoints/main/best.pt`.
   Checkpoint sha256: 3ce1f4654ac1b86aa1485a1400d1509f23c9f296fa8fc6ad6158f7831f3f93a9 (verify with `sha256sum best.pt`).
4. The face detector model (OpenCV Zoo YuNet, sha256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4)
   downloads automatically on first use.

## Result fields
| field | meaning |
|---|---|
| status | "OK" or "NO_FACE" |
| prediction | "REAL", "MANIPULATED" or null (NO_FACE) |
| confidence | softmax probability of the predicted class (uncalibrated; NOT forensic certainty) |
| real_probability, fake_probability | softmax probabilities of the two classes (fake_probability = MANIPULATED class) |
| logits | raw model outputs [REAL, MANIPULATED] |
| model_name, model | EfficientNet-B4; checkpoint file name and sha256, epoch, input size, training data |
| input_size, face_found, face_box, face_score | original image size; largest detected face as [x, y, w, h] in original pixels |
| crop_box, native_crop_side | square crop [left, top, right, bottom] in original pixels and its side length |
| warnings | list of {code, message}: no_face_detected, small_face (native crop under 96 px) |
| limitations | static list of known limits; show it with the result |

## Preprocessing (identical to training, except one step)
EXIF rotation -> RGB -> largest face (YuNet, score >= 0.6) -> square crop 1.3x the face box -> 380x380 bicubic
-> JPEG re-encode q95 -> ImageNet normalisation. Difference: training matched colour mode within each real/fake group;
a single image cannot be matched.
For large inputs face detection runs on a copy whose longer side is at most 1280 px (box mapped back; the crop
is taken from the full-resolution image), so very small faces in very large photos may be missed.

## For Part 2 (localization, robustness, video)
- `det.prepare(image)` returns (tensor, info, crop): the exact normalised input the model sees, and the crop.
- `det.detector.model` is the timm EfficientNet-B4 (eval mode). Choosing and justifying the attribution layer is Part 2's job.
- Map attribution maps (380x380 over `crop_box`) back to the original image using `crop_box`.
- Video: call analyze on frames; the detector has no temporal logic.

## For Part 3 (LLM, backend, frontend)
- Never present `confidence` as the probability that the image is fake. Suggested wording: "The detector assigned a
  fake-class probability of 0.91."
- Always show `limitations`. Handle `status == "NO_FACE"` explicitly (no prediction exists).
- Do not state certainty beyond the evidence.

## Measured performance (details in docs/experiment_log.md)
- In-domain DFF test (n=11,344): ROC-AUC 0.9974, balanced accuracy 0.9774, false-positive rate 1.45%.
- Recall by manipulation (DFF test): inpainting 98.0%, InsightFace swap 94.0%, text2img 98.8%.
- Faces under about 122 px native: recall 92.5% (versus about 98% for larger faces).
- OpenFake held-out generators (n=1,300; 90 real): ROC-AUC 0.418, recall 23.5%, false-positive rate 24.4%.
  => NOT reliable outside the training distribution.
- Not yet measured: calibration, 140k Real and Fake Faces, video, robustness (Part 2).

## Speed (measured on the dev laptop CPU, after warm-up, 3 repeats; noisy)
- Start-up: a whole command took 10-17 s including importing the libraries; loading the model and hashing the checkpoint took 1.3 s.
  Create the detector once per process and reuse it.
- Per image (detect + EfficientNet-B4 forward): about 0.3-0.5 s for a 380 px face crop (two identical-path runs gave 0.28 s and 0.52 s).
- 6000x4004 photo, face detection only: 0.63 s with the default downscaled detection (detect_max_side=1280), 1.65 s on the full-resolution image.
