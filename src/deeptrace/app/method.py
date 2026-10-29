"""About the method: fixed facts that let Q&A answer general questions ("what is evidence stability?").

Written only from the project docs, not from outside knowledge; each fact names its source. These facts describe
how DeepTrace works in general. They are never a result for a particular file, so Q&A states file results from that
file's evidence sections and uses these only to explain terms. Update them when the docs change.
  docs/detector_api.md            detector, preprocessing, labels, measured results
  docs/part2_evidence_schema.md   fields, evidence strength, stability formula and bands, video, measured context
  configs/part2_forensics.yaml    the 12 perturbations, video sampling (2 frames per second)
  src/deeptrace/forensics/evidence.py  the localization limitation (12 x 12 grid, about 32 px per cell)
"""
from __future__ import annotations

from typing import Tuple

KEY = "method"
TITLE = "About the method"
FACTS: Tuple[str, ...] = (
    # docs/detector_api.md
    "The detector is an EfficientNet-B4 image classifier (version v2), trained on DeepFakeFace and a face-cropped "
    "subset of OpenFake. It analyses only the largest detected face, cropped and resized to 380 x 380 pixels.",
    "The detector outputs a fake-class probability: its softmax probability for the MANIPULATED class. The prediction "
    "is MANIPULATED when the fake-class probability is 0.5 or more, otherwise REAL. Confidence is the probability of "
    "the predicted class. Neither is calibrated, and neither is the probability that the media is actually "
    "manipulated.",
    # docs/part2_evidence_schema.md, evidence.py
    "Model-indicated suspicious regions come from Grad-CAM, which shows which parts of the face crop pushed the "
    "detector towards MANIPULATED, on a 12 x 12 feature grid (about 32 px per cell). They are not ground-truth "
    "manipulated pixels, and localization accuracy is not yet measured.",
    "Evidence strength is model-indicated when the fake-class probability is 0.5 or more, weak when it is below 0.5 "
    "(the regions are then shown for transparency only), and none when no part of the face pushed the decision "
    "towards MANIPULATED.",
    # configs/part2_forensics.yaml, docs/part2_evidence_schema.md
    "The robustness tests re-run the detector on 12 mildly changed copies of the input: JPEG re-saving at 3 "
    "qualities, 2 downscales, 2 blur levels, 2 brightness and 2 contrast changes, and 1 noise level. They record "
    "whether the prediction flipped or the face was lost.",
    "The Evidence Stability Score is a project-defined 0 to 1 summary of how the evidence held up in the robustness "
    "tests: 0.4 x prediction stability + 0.3 x confidence stability + 0.3 x localization stability. Its bands are "
    "HIGH at 0.80 or more, MEDIUM at 0.60 or more, otherwise LOW. Any prediction flip or lost face caps the band at "
    "MEDIUM. It is not a probability that the media is manipulated; the weights and bands are design choices, not "
    "calibrated.",
    "Prediction stability is the share of the perturbed runs that kept the label. Confidence stability is the share "
    "of the distance to the 0.5 boundary that was kept. Localization stability is the mean correlation of the "
    "perturbed attribution maps with the original map. Higher is more stable for all three.",
    "For videos, frames are sampled at 2 per second and analysed one by one; there is no temporal model. The video "
    "score is the mean fake-class probability over frames with a usable face. A suspicious interval is at least 2 "
    "sampled frames in a row with fake-class probability 0.5 or more. The video Evidence Stability Score is the "
    "median over 3 key frames: the two most suspicious frames plus the median frame.",
    "Temporal stability is the share of consecutive face-frame pairs with the same label and a change in fake-class "
    "probability under 0.2.",
    "NO_FACE means no face was detected, so the detector made no prediction and no forensic checks were run; there "
    "is no verdict.",
    # docs/part2_evidence_schema.md "Measured context"
    "Measured results: ROC-AUC 0.9966 on DeepFakeFace test data, 0.82 on held-out OpenFake generators and 0.49 "
    "(chance) on StyleGAN faces, so the detector does not generalise to every generator. Video accuracy and "
    "localization accuracy are not yet measured.",
    # docs/part3_llm_setup.md
    "The explanations and answers are written by a local language model (qwen3:4b-instruct) from these facts only "
    "and checked by fixed rules against the evidence; the language model is not a detector.",
)
