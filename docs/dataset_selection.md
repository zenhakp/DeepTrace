# Dataset selection (Part 1)

Requirement (team decision): the dataset must be directly obtainable and usable without owner
approval, access forms or waiting for emailed links, with clear terms, real + fake classes and
reliable labels. FaceForensics++ and Celeb-DF are therefore NOT primary datasets (gated).

## Chosen
- **Primary (train / validation / in-domain test): DeepFakeFace (DFF)**, `OpenRL/DeepFakeFace`
  on Hugging Face, pinned to revision `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c`.
  Checked via the Hub API: `gated: False`.
- **External test only: OpenFake**, `ComplexDataLab/OpenFake`, pinned to
  `3247b942f18d39d1fd476b1077852af033d35d8c` (`gated: False`, license metadata cc-by-nc-4.0).
  A face-filtered subset will be chosen later. Never used for training.
- **Small external set (to be verified): 10,000 Real vs Fake Faces (StyleGAN3)** on Kaggle
  (stated license CC BY-NC-SA 4.0).

## Rejected (gated or terms not verified)
Celeb-DF, DF40, WildDeepfake, FakeAVCeleb, DeepSpeak (all require a form/agreement);
DFDC (per-user AWS account, license text NOT VERIFIED); Hugging Face re-uploads of FF++ or DF40
(re-hosting of gated data); CIFAKE (32x32 images, unsuitable for EfficientNet-B4).

## Open items
- DFF license: Hub metadata says openrail, card body says apache-2.0, GitHub says Apache-2.0.
  Real images come from IMDB-WIKI, which states academic research use only. Treat as
  academic-only. Redistribution/sharing terms: NOT VERIFIED. Each member downloads their own copy.
- Dataset sizes and class counts as downloaded by us: Not yet measured.
- DFF has a 1:3 real:fake ratio and fakes are derived from real images (paired); the split is
  by celebrity identity to avoid leakage (decided at the splitting step).
- No ungated labelled face-swap video dataset was verified; implications for Part 2 are open.
- Cross-dataset drops mix generator shift and real-image-source shift (limitation to report).

## Measured findings and corrections (Steps 4-5)
Measured by our inspection of the pinned download (DeepFakeFace, revision eb2a54fb):
- 30,000 images in each of wiki (real), inpainting, insight, text2img zips: 120,000 images, 1:3 real:fake.
- Every real name appears once in each fake zip with the same relative path, so a "group" is one real
  photo plus its three fakes.
- Each filename ID occurs once; filenames do not identify a person beyond the photo. This SUPERSEDES the
  earlier statement that the split would be by celebrity identity: the split unit is the group.
  Limitation: the same person under two IDs cannot be detected and could cross splits.
- In a 200-image sample per zip, all fakes are 512x512 RGB, while real images vary in size (154
  distinct sizes) and about 10% are grayscale. Full-dataset measurement: see the split report from
  scripts/build_manifest.py. Image size/colour mode could act as a shortcut; preprocessing must
  neutralise it (decision pending), and the size-only rule baseline is reported next to detector results.

## Shortcut finding and preprocessing design (Step 6)
Measured on all 120,000 images: real images are never exactly 512x512 (98.25% non-square, 9.47%
grayscale); every manipulated image is exactly 512x512 RGB. A rule "predict MANIPULATED iff 512x512"
scores accuracy/precision/recall 1.0 on train, validation and test. Raw-image accuracy is therefore
meaningless. Mitigation (design choices): face crop with identical geometry and resize for all images,
group-level colour-mode matching, identical JPEG re-encoding, group kept only if a face is found in all
its images. Remaining shortcuts are measured by the metadata audit in the preprocessing report
(native crop size, face area ratio, JPEG quantization, bytes/pixel, chroma, sharpness).
A deliberately uncontrolled "naive" pipeline will be run once as an ablation to show shortcut learning.

## Preprocessing results (measured, full run, laptop CPU, about 1h49m)
Settings: YuNet detector (score >= 0.6), square crop 1.3x face box, 380x380, bicubic, JPEG q95,
colour-mode matching per group, a group kept only if a face is found in all 4 images.
- Face detection rate: real 98.81%, inpainting 98.52%, insight 98.48%, text2img 98.85%.
- Groups kept: train 22,759/24,000, val 2,848/3,000, test 2,836/3,000 (28,443 total).
  Kept images: 28,443 real and 85,329 manipulated (ratio exactly 1:3).
- Median native crop side: real 160 px, manipulated 212 px.
- Single-feature shortcut audit (strength = max(AUC, 1-AUC); 0.5 = no information):
  sharpness 0.502, mean_chroma 0.541, face_area_ratio 0.571, native_crop_side 0.626,
  bytes_per_px 0.868, q_lum_mean (source JPEG quantization) 0.981.
- Interpretation: q_lum_mean and bytes_per_px describe the source files before our re-encoding. They show
  that real and fake images have different compression histories. Whether traces survive the crop,
  resize and re-encode is NOT yet measured.
- Defences (design choices): class-independent degradation augmentation in training; a matched
  evaluation subset (q_lum_mean distribution equalised across classes); pair-matched re-preprocessing
  kept as a possible later ablation. Metrics are reported on the full test split AND the matched subset.
