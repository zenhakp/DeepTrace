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
