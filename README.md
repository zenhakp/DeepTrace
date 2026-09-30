# DeepTrace

Intelligent Deepfake Detection and Forensic Analysis (final-year project, 3 members).

Layout (packages are added as each part needs them):

    src/deeptrace/common/     shared foundation (paths, config, device, seed)  <- Part 1, step 1
    src/deeptrace/detection/  Part 1 (dataset, detector, training, evaluation)
    src/deeptrace/forensics/  Part 2 (planned)
    src/deeptrace/app/        Part 3 (planned)

Data, checkpoints and results live OUTSIDE git, under a root directory set by the
`DEEPTRACE_ROOT` environment variable (on Colab: a folder in your mounted Google Drive).
Default when unset: `./deeptrace_data`.

Setup: `pip install -e . && pip install pytest` then `pytest`.
