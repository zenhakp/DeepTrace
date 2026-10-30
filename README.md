# DeepTrace

Intelligent Deepfake Detection and Forensic Analysis (final-year project, 3 members).

Layout:

    src/deeptrace/common/     shared foundation (paths, config, device, seed)          Part 1
    src/deeptrace/data/, models/, training/, evaluation/   detector and dataset       Part 1
    src/deeptrace/forensics/  localization, robustness, video, structured evidence     Part 2
    src/deeptrace/app/        evidence checks, LLM explanation, Q&A, FastAPI backend   Part 3
    frontend/                 Next.js web UI                                           Part 3

Data, checkpoints and results live OUTSIDE git, under a root directory set by the
`DEEPTRACE_ROOT` environment variable (on Colab: a folder in your mounted Google Drive).
Default when unset: `./deeptrace_data`.

Setup: `pip install -e . && pip install pytest` then `pytest`.

## Part 3: run the app locally (from a clean clone)

The app runs on one laptop, on CPU, with no paid API and no Colab. Tested on Windows 11 with Git Bash. Details,
design decisions and limitations: [docs/part3_app.md](docs/part3_app.md). Demo steps:
[docs/part3_demo_checklist.md](docs/part3_demo_checklist.md).

**Needs:** Python 3.10, Node.js 20.9 or newer, roughly 1.5 GB of disk for the Python and Node packages (an
estimate; CPU PyTorch is most of it), and the private detector checkpoint (68 MB, from the team's Drive; never
commit it). The explanation model is optional and needs about 7 GB more (see step 4).

1. Python environment (repository root):

       py -3.10 -m venv .venv
       .venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       .venv/Scripts/python.exe -m pip install -r requirements.txt
       .venv/Scripts/python.exe -m pip install -e .

2. Detector checkpoint: put `best.pt` at `<DEEPTRACE_ROOT>/checkpoints/mixed/best.pt` and set `DEEPTRACE_ROOT`
   (for example `export DEEPTRACE_ROOT="$HOME/deeptrace_data"` in `~/.bashrc`). Check it with
   `sha256sum "$DEEPTRACE_ROOT/checkpoints/mixed/best.pt"`; expected
   `d00ae780e23a7f713d3f7af8e57e05a4a5e08382c005772499e9a7fcedabd107` ([docs/detector_api.md](docs/detector_api.md)).

3. Frontend packages (once, about 380 MB in `frontend/node_modules`):

       cd frontend
       npm ci
       npx next telemetry disable
       cd ..

4. Optional, the explanation model: install Ollama without start at login and pull `qwen3:4b-instruct`, as in
   [docs/part3_llm_setup.md](docs/part3_llm_setup.md) (which also has the stop and uninstall steps). Without it
   the app still works: every report is shown as evidence text.

5. Start everything with one command (repository root):

       .venv/Scripts/python.exe scripts/run_app.py --with-llm

   Open http://127.0.0.1:3000. Leave out `--with-llm` to run without the explanation model (an Ollama that is
   already running is used either way). The first start builds the frontend (about 30 s). Server output goes to
   `~/Downloads/deeptrace_outputs/app_logs/`; uploads and results to `~/Downloads/deeptrace_outputs/app_jobs/`
   (outside the repository).

6. Stop: press **Ctrl+C** in that terminal; it stops everything the script started. If a server is ever left
   running (for example after closing the terminal window), run
   `.venv/Scripts/python.exe scripts/run_app.py --stop`.

7. Tests:

       .venv/Scripts/python.exe -m pytest -q
       cd frontend && npm test && npm run build
