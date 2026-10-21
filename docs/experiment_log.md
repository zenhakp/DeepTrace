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

## Cross-dataset: OpenFake test shards (measured)
- Source: ComplexDataLab/OpenFake revision 3247b942, core/test shards 0 and 6 of 13, deterministic 30% sample
  (4,199 images: 2,126 real / 2,073 fake), same face-crop pipeline (YuNet, margin 1.3, 380 px, q95), min native crop side 96.
- Face filter: 4.2% of real images kept (90) vs 58.4% of fake images (1,210). The real class is a small, atypical subset.
- Model: "main" best.pt (epoch 3), threshold 0.5, no retraining. n=1,300 (90 real / 1,210 manipulated).
- ROC-AUC 0.418; balanced accuracy 0.495; recall 0.235 (284/1,210); specificity 0.756 (68/90; 95% interval about 0.66-0.83);
  false-positive rate on real 24.4% (in-domain 1.45%). Accuracy 0.271 is below the trivial "always MANIPULATED" rate (0.931).
- Recall by generator (n >= 75): flux-2-klein-9b 3.6% (306), gpt-image-1.5 17.0% (135), midjourney-7 16.0% (75),
  z-image-turbo 27.3% (388), illustrious 32.4% (111). Generators with n < 60 not interpreted.
- Conclusion: the detector does not generalize to these held-out modern generators (in-domain AUC 0.997 vs cross-dataset 0.42).

## Positive control for the external-data path (measured)
- 400 DFF test crops (200 real / 200 fake, random, seed 0) went through scripts/preprocess_external.py as if they were a
  foreign dataset, then were evaluated with "main" best.pt.
- Kept 391 (198 real / 193 fake). ROC-AUC 0.9960, accuracy 0.9719, balanced accuracy 0.9718, recall 0.9689, FPR 0.0253 (5/198).
- Conclusion: the external path (re-crop, re-encode at q95, labels, evaluation) preserves in-domain performance
  (in-domain test AUC 0.9974), so the OpenFake result reflects domain shift, not a pipeline fault.
- Limit: these crops were already face-centred at 380 px, so it does not exercise cropping from large images or other face
  scales (OpenFake native crop sides, median 225-275 px, lie within the range seen in training).
- Not yet measured: 140k Real and Fake Faces, any run trained on more generators, calibration.

## Feasibility probe: OpenFake TRAIN shard (measured, 2026-10-20)
- Shard core/train-00016-of-00032-00000 (5.73 GB), 50% deterministic sample, same face pipeline: 2,040 images.
- Real faces kept 484/1,031 (46.9%), fake faces kept 629/1,009 (62.3%). The test shards had 4.2% real / 58.4% fake.
- About 968 real + 1,258 fake faces per full shard (estimate from the 50% sample). 608 train shards, 3.3 TB in total.
- 69 generators in this one shard. Test generator families overlap train as different versions (flux-2, gpt-image,
  midjourney, seedream, ideogram, wan), so OpenFake-test gains after training on OpenFake-train must be reported by family.
- Decision: experiment 2 = same recipe from ImageNet on DFF + OpenFake-train faces (6 train + 1 val shard); evaluate on
  DFF test, OpenFake test (by family), 140k, and the positive control; report both runs.

## OpenFake train subset for experiment 2 (measured, built 2026-10-20)
- 7 evenly spaced shards of the train split (6 train + 1 val, by whole shard), same face pipeline, min native crop side 96.
- 27,925 images processed, 15,232 kept (54.5%). Train: 5,658 real + 7,420 manipulated. Val: 931 real + 1,223 manipulated.
- 74 generators; median native crop side manipulated 223 px, real 283 px.
- Face-cropping took about 5,000-5,300 s per full shard on a Colab CPU (about 1.3 s per image), about 7.8 h in total.
- Planned mixed training: class-balanced within each source (DFF 22,759+22,759, OpenFake 5,658+5,658 per epoch),
  3,553 steps per epoch, 10,659 steps in total; checkpoint chosen by the mean of per-source validation ROC-AUC.
