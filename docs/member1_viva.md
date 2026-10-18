# Member 1 (Part 1) viva notes: detection, dataset, generalization

## One-minute summary
We needed a deepfake dataset usable without approval forms, so we used DeepFakeFace (Hugging Face, pinned revision),
audited it, found it was trivially separable by image size (a one-line rule scored 1.0), neutralised that with a
uniform face-crop pipeline, trained an ImageNet-initialised EfficientNet-B4 (test ROC-AUC 0.997), and then showed
honestly that it does not generalize to modern held-out generators (ROC-AUC 0.42 on an OpenFake sample).

## Concepts (what / why / how / limitation)
1. Dataset selection without gated access. Gated sets (FF++, Celeb-DF, DF40...) need approval, so we chose
   DeepFakeFace (30,000 real + 90,000 fake, three manipulation types) after checking its source and license. Limitation:
   license metadata is inconsistent (openrail / Apache-2.0) and real images come from IMDB-WIKI (academic use only);
   sharing the model outside the team is not assumed to be allowed.
2. Leakage and group splitting. Each real photo has three fakes derived from it, so splitting by image would put a
   photo in train and its fake in test. We split by group (a real photo plus its fakes), 80/10/10, using a seeded hash.
   Limitation: the same person under two IDs cannot be detected from filenames.
3. Shortcut learning. Fakes were all 512x512 RGB, reals were not: a size rule scored accuracy 1.0. We used identical
   face crops, group colour matching and identical JPEG re-encoding, then audited residual cues. Source JPEG tables still
   differed (fakes about quality 75, most reals about 80). Limitation: we can only audit features we thought to measure.
4. EfficientNet-B4 and transfer learning. A CNN scaled in depth, width and resolution; we fine-tune all layers from
   ImageNet weights (AdamW, lr 2e-4, cosine schedule). Limitation: ImageNet features are not forensic features and
   nothing was tuned.
5. Class-balanced epochs and degradation augmentation. Each epoch uses all real images plus an equal number of fakes
   (re-drawn each epoch); blur, resize and JPEG are applied to every image regardless of class. Limitation: fakes are not
   all seen every epoch.
6. Metrics. ROC-AUC (threshold-free ranking quality), balanced accuracy (fair under 1:3), precision, recall, F1,
   confusion matrix, plus recall per manipulation type. Threshold 0.5 is fixed in advance; the test set was evaluated once.
7. Matched subset and within-class analysis. Matching on the source JPEG table gave only 154 test images (small
   sample); within-class bins showed no rise in false positives toward fake-like compression. Limitation: observational.
8. Cross-dataset generalization and the positive control. Same checkpoint, same preprocessing, no retraining, on OpenFake
   test shards (held-out generators). A control (unseen DFF crops through the same external path) kept AUC at 0.996, so
   the failure is domain shift, not a pipeline bug. Limitation: the control did not exercise other face scales.
9. Detector API. DeepTraceDetector returns a structured, JSON-serialisable result with warnings and limitations; no face
   means no prediction. Confidence is a softmax probability, not forensic certainty.

## Numbers to remember (all measured; see docs/experiment_log.md)
- DFF: 120,000 images, 30,000 groups; 28,443 groups kept after the face filter (94.8%).
- Training: 8,535 steps, about 2.56 it/s on a Colab T4, about 18.5 min per epoch.
- Validation ROC-AUC 0.9969; test (n=11,344) ROC-AUC 0.9974, balanced accuracy 0.9774, false-positive rate 1.45%.
- Recall by type (test): inpainting 98.0%, InsightFace swap 94.0%, text2img 98.8%; faces under about 122 px: 92.5%.
- Matched subset: n=154. OpenFake: n=1,300 (90 real), ROC-AUC 0.418, recall 23.5%, false-positive rate 24.4%.
- Control: ROC-AUC 0.996 (n=391).

## Likely questions, honest answers
- Why did it not generalize? Untested hypotheses: it learned fingerprints of DFF's generators (Stable Diffusion 1.x,
  InsightFace) that 2025 generators do not have, and/or DFF's real-versus-fake differences do not carry over. A second
  run with more generator diversity would test the first.
- Is 0.997 good? In-domain only. It says the model learned this dataset, which has known cue differences.
- Why not FF++? Gated; the project requirement was approval-free data.
- What is confidence? A softmax probability, uncalibrated; not the probability the image is fake.
- What would you do next? Train with more generators, evaluate on 140k Real and Fake Faces, calibrate, hand attribution
  and stability analysis (Part 2) the stable API.
- What is not yet measured? Calibration, 140k set, video, robustness, any run trained on more generators.