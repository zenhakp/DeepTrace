# Part 3 demo checklist

## The day before
- [ ] `git pull` on the branch you will demo; `.venv/Scripts/python.exe -m pytest -q` and
      `cd frontend && npm test && npm run build` pass.
- [ ] `sha256sum "$DEEPTRACE_ROOT/checkpoints/mixed/best.pt"` matches docs/detector_api.md.
- [ ] Ollama installed, `qwen3:4b-instruct` pulled (`ollama list`), start at login still disabled.
- [ ] At least 5 GB free on C:.
- [ ] Demo files ready OUTSIDE the repository (for example `~/Downloads/samples`): one image the detector calls
      MANIPULATED (MEDIUM stability is a good story, for example `2.jpg`), one image it calls REAL, one short video,
      one image without a face.
- [ ] Do one full dry run of the flow below; it also warms up the model.

## Start (about 1 minute before)
- [ ] Close other heavy programs (the LLM needs about 3 GB of RAM).
- [ ] `.venv/Scripts/python.exe scripts/run_app.py --with-llm`, wait for "DeepTrace is running".
- [ ] Open http://127.0.0.1:3000; the status line shows "Detector checkpoint found" and the explanation model as
      available.

## Flow (about 8 minutes)
1. Upload page: drop the MANIPULATED image; the drop area is replaced by the file card; Analyse.
2. Results: the steps move from Analysis to Evidence report in about 15 to 30 s. Point out the summary strip,
   the "not a probability" note, the overlay (click to enlarge) and the "not ground-truth" note, the Evidence
   Stability bars, the robustness flips.
3. "Generating explanation..." for about 80 s: the report is already complete as evidence text. When it switches,
   open "Evidence for this section" to show that every sentence is backed by facts and was checked.
4. Questions, one at a time:
   - "Why is the Evidence Stability MEDIUM?" (grounded answer)
   - "why?" (a follow-up that uses the previous answer)
   - "What is evidence stability?" (from the About-the-method facts)
   - on a HIGH file, "why is the evidence stability only medium" (immediate correction from the facts)
   - "Who is this person?" (refused, with suggested questions; click one)
5. Video: the timeline with the suspicious interval, the key-frame overlays.
6. Image without a face: "No face found, no verdict" everywhere; no regions section.
7. Download the Markdown report.
8. Limitations card: read one measured limitation aloud (for example chance level on StyleGAN faces).

## If something goes wrong
- The LLM is slow or not running: the report is complete without it ("Language model unavailable" or evidence
  text). Carry on; this is the designed fallback.
- "Port 8000 (or 3000) is already in use": `.venv/Scripts/python.exe scripts/run_app.py --stop`, then start again.
- Upload fails with `checkpoint_missing`: DEEPTRACE_ROOT is not set in that terminal (`echo $DEEPTRACE_ROOT`).
- Logs: `~/Downloads/deeptrace_outputs/app_logs/backend.log` and `frontend.log`.

## Stop
- [ ] Ctrl+C in the terminal running `run_app.py`.
- [ ] `tasklist //FI "IMAGENAME eq ollama.exe"` and `tasklist //FI "IMAGENAME eq node.exe"` show nothing left.

## Say clearly
- The detector is the source of truth; the language model only explains its evidence and is checked by fixed
  rules.
- Scores are uncalibrated detector outputs, not probabilities that the media is manipulated.
- Regions are model-indicated, not ground-truth manipulated pixels; localization and video accuracy are not yet
  measured.
