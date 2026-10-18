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

## Shortcut analysis on the test predictions (measured)
- Source JPEG luminance-table mean (q_lum_mean): every manipulated test image has 29.031; at least 60% of real
  test images have 23.078 (about JPEG quality 80, versus about 75 for the fakes; my inference from standard tables).
- The q-matched test subset (n=154) is exactly the 77 real test images whose table equals the fakes' plus 77 fakes.
  On it accuracy was 0.987 (152/154), ROC-AUC 0.9992. Small sample (wide interval), but the cue is identical in both classes.
- Within-class analysis (5 equal-count bins): real-image false-positive rate does not rise toward the fakes' table
  value (2.0%, 1.8%, 0.4%, 1.3%, 2.0%; noise of about +-1 point). No evidence of reliance on this cue; not proof of absence.
- Smallest faces are the weak spot: fake recall 92.5% for native crop side under 122 px vs 97.5-98.5% above;
  real-image FPR 2.3% for the smallest bin vs 0.7-1.8% above.
- Crops are re-encoded at q95 after a crop and non-integer resize, which should break JPEG block alignment (reasoning, not measured).
EOFcat >> docs/experiment_log.md << 'EOF'

## Shortcut analysis on the test predictions (measured)
- Source JPEG luminance-table mean (q_lum_mean): every manipulated test image has 29.031; at least 60% of real
  test images have 23.078 (about JPEG quality 80, versus about 75 for the fakes; my inference from standard tables).
- The q-matched test subset (n=154) is exactly the 77 real test images whose table equals the fakes' plus 77 fakes.
  On it accuracy was 0.987 (152/154), ROC-AUC 0.9992. Small sample (wide interval), but the cue is identical in both classes.
- Within-class analysis (5 equal-count bins): real-image false-positive rate does not rise toward the fakes' table
  value (2.0%, 1.8%, 0.4%, 1.3%, 2.0%; noise of about +-1 point). No evidence of reliance on this cue; not proof of absence.
- Smallest faces are the weak spot: fake recall 92.5% for native crop side under 122 px vs 97.5-98.5% above;
  real-image FPR 2.3% for the smallest bin vs 0.7-1.8% above.
- Crops are re-encoded at q95 after a crop and non-integer resize, which should break JPEG block alignment (reasoning, not measured).
