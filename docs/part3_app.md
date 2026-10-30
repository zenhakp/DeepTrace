# Part 3: evidence-grounded explanation, backend and web app

Part 3 turns Part 2's structured evidence JSON (schema 1.0) into something a person can read and question, without
adding anything the evidence does not contain. The detector (Part 1) is the source of truth; the language model is
not a detector and only explains the evidence it is given. Everything runs locally on CPU. How to install and run:
[README](../README.md#part-3-run-the-app-locally-from-a-clean-clone). Demo: [part3_demo_checklist.md](part3_demo_checklist.md).
Local LLM setup, stop and uninstall: [part3_llm_setup.md](part3_llm_setup.md).

## Flow

    upload (Next.js, :3000) -> FastAPI (:8000) -> scripts/analyze.py in a subprocess (Parts 1 and 2, unchanged)
      -> <name>_evidence.json + overlay PNGs in the job folder
      -> load_evidence(): JSON Schema + cross-field checks          (evidence.py)
      -> digest: fact sentences per report section                  (digest.py)
      -> template report, shown at once                             (explain.py)
      -> local LLM writes one short text per section                (llm.py, Ollama, qwen3:4b-instruct)
      -> deterministic checks against the facts; failures fall back to the facts  (validate.py)
      -> questions: same digest + "About the method" + last checked answers       (qa.py, method.py)
      -> Markdown report                                            (report.py)

| file | purpose |
|---|---|
| `src/deeptrace/app/evidence.py` | load and check one evidence document (schema 1.0 plus rules the schema cannot express) |
| `src/deeptrace/app/digest.py` | the evidence as fact sentences per section: shown facts and LLM facts |
| `src/deeptrace/app/llm.py` | standard-library client for Ollama on 127.0.0.1 with stable failure codes |
| `src/deeptrace/app/explain.py` | grounded explanation, one text per section, template fallback |
| `src/deeptrace/app/validate.py` | deterministic rule checks of LLM text against the facts |
| `src/deeptrace/app/qa.py`, `method.py` | grounded Q&A: premises, method facts, history, refusals with suggestions |
| `src/deeptrace/app/report.py` | Markdown report |
| `src/deeptrace/app/pipeline.py` | runs `scripts/analyze.py` as a subprocess |
| `src/deeptrace/app/jobs.py` | job folders, `job.json` records, analysis thread and LLM thread |
| `src/deeptrace/app/api.py` | FastAPI routes |
| `frontend/` | Next.js upload page and results page |
| `scripts/run_app.py` | one-command start and stop |
| `configs/part3_app.yaml` | LLM and app settings |

API (all JSON errors are `{"detail": {"code", "message"}}`): `GET /api/health`, `POST /api/jobs` (202),
`GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/files/{name}` (listed overlays only),
`GET /api/jobs/{id}/report`, `POST /api/jobs/{id}/questions` (202). No request waits for the LLM: the job state,
`explanation_state` and each question's state are polled.

## Design decisions

Each: decision, alternatives, why, trade-off.

1. **Validate evidence with `jsonschema` plus cross-field rules.** Alternatives: Part 2's built-in validator,
   Pydantic models. Why: a full draft 2020-12 validator without importing Part 2 code (which loads torch), and no
   second copy of the schema; the schema cannot express "NO_FACE has no verdict", so that is checked in code.
   Trade-off: the cross-field rules follow Part 2's current behaviour and need updating if it changes without a
   `schema_version` change.
2. **Local LLM: Ollama with `qwen3:4b-instruct`.** Alternatives: `llama3.2:3b` (measured), `llama-cpp-python`,
   Hugging Face `transformers` on CPU, paid APIs (ruled out). Why: one installer, a local HTTP API, 4-bit models in
   about 3 GB of RAM; qwen made no real wording violation on the three examples while llama stated scores as
   certainty twice and invented a NO_FACE verdict (part3_llm_setup.md). Trade-off: about twice as slow as llama.
   Plain `qwen3:4b` is not used: that tag is the slower "thinking" variant.
3. **The LLM sees a digest, not the JSON.** Facts per section: the shown facts quote Part 2's sentences unchanged;
   the facts sent to the LLM are phrased so they cannot be misread (regions by rank and position only; stability
   values and points lost as separate sentences), after both were misread in a live run. Limitations and warnings
   are never sent and always shown verbatim (schema rule 6), which also cut the time per explanation by about 39%.
   Probabilities never round to look certain (0.9994 is "above 0.99").
4. **One short text per section (JSON output).** Alternative: one paragraph. Why: each section is checked on its own
   and falls back on its own. Trade-off: more output tokens.
5. **Deterministic validation, no second LLM.** Rules: no verdict on NO_FACE, banned certainty words (the required
   "not ground-truth manipulated pixels" is allowed), percentages as certainty, numbers and labels not in the cited
   facts, verdict claims, probabilities given as 1 or 0, any probability in the regions section, cost clauses whose
   number is not a points-lost value, claims of a label the detector did not give. Why: repeatable, instant,
   testable; every rule comes from a real failure (tests/data/llm_known_failures.json) and must keep real accurate
   answers passing (tests/data/llm_known_good.json). Trade-off: patterns, not meaning (see limitations).
6. **Q&A as a short, grounded conversation.** One question at a time per file; the last 3 checked answers are
   context for follow-ups; refusals never enter the history. A fixed "About the method" section, written only from
   the project docs, answers general questions. False premises are corrected from the facts: a wrong Evidence
   Stability band at once (band words are precise), a wrong prediction when the LLM refuses (words like "real" also
   occur in ordinary questions). Identity questions are refused without an LLM call. When a question is outside
   the analysis, the answer offers three questions this file can answer.
7. **The backend runs `scripts/analyze.py` as a subprocess.** Alternative: import the pipeline in the API process.
   Why: exactly the code path Part 2 tested and schema-checked, no copy of Part 2 code, torch memory is freed before
   the LLM runs, a crash cannot take the API down. Trade-off: about 10 to 15 s of start-up per analysis.
8. **Two worker threads, no task queue.** The analysis thread and the LLM thread each do one thing at a time (both
   are CPU heavy; Ollama serves one request at a time). The report is shown as soon as the analysis is done; the LLM
   text and answers arrive later. Alternatives: Celery/Redis (ruled out), FastAPI background tasks. Trade-off: an
   analysis and an LLM call can run at the same time and slow each other down.
9. **Persistence: one `job.json` per job folder, no database.** The folder already holds the upload, the evidence
   and the overlays. Records are rewritten atomically on every change and re-read on start-up; evidence files are
   checked again rather than trusted. Interrupted work is not resumed (nothing CPU heavy starts by itself).
   Alternative: SQLite. Trade-off: no queries across jobs.
10. **Next.js with plain CSS and no UI, chart or test libraries.** `/api/*` is forwarded to the backend, so the
    browser sees one origin. The display rules (NO_FACE never shows a label, probabilities never round to 1.00,
    MANIPULATED amber and REAL slate, never red or green) are plain functions with tests. System fonts, so a
    rebuild needs no internet. Next 16.4.3, because 16.0.0 to 16.4.0 have published security advisories; telemetry
    disabled.
11. **Local only.** Both servers listen on 127.0.0.1; upload names are reduced to safe characters; only overlay
    files listed in the evidence are served; outputs live outside the repository. No accounts or login (out of
    scope).

## Viva notes

- **LLM grounding.** What: limiting the model to the facts it is given. Why: the detector is the source of truth.
  How: the evidence becomes fact sentences; the prompt holds only those facts and fixed rules; the answer must be
  JSON with one text per section. Why chosen: the same facts are the fallback report, so the app shows the same
  evidence with or without the LLM. Limitation: it lowers, but does not remove, the chance of misplaced facts.
- **Evidence validation.** What: fixed rules that compare each LLM sentence with the facts it was given. Why: a 4B
  model can move a fact to the wrong place or overstate it. How: numbers, labels, banned wording and verdict
  patterns are checked against the cited sections; any problem shows the facts instead, with the reason. Why
  chosen over a second LLM: repeatable, instant on CPU, testable. Limitation: patterns, not meaning.
- **FastAPI.** What: a Python web framework; each function is an HTTP endpoint with request checking. Why: the UI
  needs upload, status and results. How: uvicorn serves it; slow work runs on two threads, so requests return at
  once. Limitation: one local user; records are in memory with `job.json` as the saved copy.
- **Next.js.** What: a React framework; each folder under `app/` is a page. Why: the project specifies it, and it
  gives routing plus forwarding to the backend in one config file. How: both pages run in the browser and poll the
  backend every 2 s while something is running. Limitation: the largest dependency (about 380 MB) and it needs
  patch upgrades when advisories appear.
- **Persistence.** What: one JSON record per job folder. Why: a restart should not lose results that cost minutes
  of CPU. How: atomic rewrite on every change, rebuilt from the folders on start-up. Limitation: no queries.
- **Grounded Q&A.** What: questions answered from one file's evidence and the method facts. Why: users ask
  follow-ups. How: the LLM must cite the sections it used or declare the question unanswerable, and its text is
  checked against those sections; false premises are corrected from the facts. Limitation: about a minute for a
  first question on CPU, and only the last 3 checked answers are remembered.

## Measured (one laptop: Ryzen 5 5500U, 15.3 GB RAM, CPU only, Windows 11; single runs, indicative)

| what | measured |
|---|---|
| image analysis in the app (subprocess including start-up), `2.jpg` | 15 to 27 s |
| 5 s video analysis | about 45 s (Part 2 measurement, `docs/part2_forensics.md`) |
| explanation, first prompt version (all sections, no length cap) | 131.0 s image, 115.6 s video |
| explanation, final version | 80.1 s image (991 prompt / 266 output tokens), 70.7 s video (1175 / 291); 82.4 s in the app |
| first question on a file | 75.1 s |
| later questions on the same file (Ollama reuses the unchanged start of the prompt) | 9.0 to 11.1 s |
| band false-premise correction, identity refusal (no LLM call) | immediate |
| question POST through the frontend proxy | 172 ms (it returns before the LLM runs) |
| model RAM while loaded (`ollama ps`) | 3.2 GB (llama3.2:3b: 2.6 GB) |
| disk: frontend packages / Ollama / model | 379 MB / at least 4 GB / 2.5 GB |
| model choice (3 example files, interim prompt) | qwen3:4b-instruct 0 real wording violations, 27.6 to 61.9 s; llama3.2:3b 3 violations, 17.0 to 28.9 s |

Tests: 458 Python tests pass (3 skipped: real-detector checks that need a face photo), 17 frontend tests pass, and
the production build type-checks every page. Live end-to-end checks with the real detector: an image and a video
through the frontend proxy, a 15 MB upload arriving complete and a 101 MB upload refused (413), job records
surviving a server restart unchanged, the start script starting and stopping both servers.

## Limitations

- The validator checks patterns, not meaning. It can reject accurate text (then the facts are shown, which is
  always correct), and it can miss a wrong statement that reuses a number from the same section.
- Deterministic premise corrections cover only the Evidence Stability band and the prediction label; other false
  premises depend on the LLM following its rules.
- The language model is small (4B) and slow on CPU: about 80 s per explanation and about 75 s for a first question.
  Questions on a file are answered one at a time and wait behind an explanation in progress.
- Explanation and answer quality were judged by reading about 20 real outputs, not measured on a labelled set.
- The "About the method" facts and the suggested questions are written by hand from the docs and must be updated
  when the docs change.
- Work interrupted by a server restart is not resumed: the analysis must be uploaded again, an interrupted
  explanation stays as evidence text, an interrupted question is marked as failed.
- One local user, no accounts. Tested on Windows 11 only; `run_app.py --stop` is Windows only.
- The frontend has no browser automation tests: its rules are tested as plain functions, plus the build and live
  checks.
- Everything Part 2 lists still applies (shown in every report): the detector does not generalise to every
  generator, mild blur and recompression can flip predictions, regions are reported for video key frames only.

## Not yet measured

- Localization accuracy (no manipulation masks) and video detection accuracy (Part 2).
- Calibration of the detector's probabilities.
- Accuracy of explanations and answers on a labelled set, and the validator's false-positive and false-negative
  rates.
- Speed on other hardware, and behaviour with more than one user at a time.
