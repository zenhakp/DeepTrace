# Experiment log (Part 1)

## Run "main": EfficientNet-B4 (ImageNet init), DeepFakeFace, 3 epochs, seed 42
- Dataset revision eb2a54fb (DeepFakeFace); face crops 380 px (s380_m1.3_q95); config: experiments/main/config.yaml (Drive).
- Train 91,036 images (22,759 real / 68,277 manipulated); class-balanced epochs (all real + equal number of fakes, re-drawn each epoch).
- Colab T4, mixed precision, about 2.56 it/s, 8,535 steps.
- Validation (n=11,392; 2,848 real / 8,544 manipulated; threshold 0.5), epochs 1/2/3:
  ROC-AUC 0.9924 / 0.9965 / 0.9969; balanced accuracy 0.9483 / 0.9747 / 0.9773. Best = epoch 3.
- Test, evaluated once on best.pt (n=11,344; 2,836 / 8,508): ROC-AUC 0.9974, accuracy 0.9734,
  balanced accuracy 0.9774, precision 0.9951, recall 0.9693, specificity 0.9855, F1 0.9820;
  TP 8247, FP 41, FN 261, TN 2795.
- Share predicted MANIPULATED by source (test): real 1.45% (false-positive rate), inpainting 98.03%,
  insight 93.97%, text2img 98.80% (recall).
- Matched test subset (q_lum_mean matched, 20 bins): n=154 (77/77), only 2.72% of real and 0.91% of
  manipulated test images. ROC-AUC 0.9992, accuracy 0.9870 (152/154), recall 77/77, FP 2/77.
  INCONCLUSIVE: tiny sample, and matching unbalanced other features (native_crop_side strength 0.785,
  sharpness 0.749).
- In-domain results only; they say nothing about generalization.
- Not yet measured: within-class shortcut analysis, cross-dataset results, pair-matched ablation, calibration.
