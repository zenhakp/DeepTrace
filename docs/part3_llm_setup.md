# Part 3 local LLM: setup, start, stop, uninstall

The explanation text is written by a small open-weights LLM running on this laptop through **Ollama**. No paid API,
no internet once the model is downloaded. The LLM is never a detector: it only rewrites the evidence JSON it is
given. When it is not running, the app still works and shows the deterministic report.

Ollama listens on `127.0.0.1:11434` only (not reachable from other machines). Settings: `configs/part3_app.yaml`.
Client: `src/deeptrace/app/llm.py`.

Policy for this project: **Ollama never runs in the background.** It does not start at login; it runs only in a
terminal you open for the app, and Ctrl+C stops it. A model stays in RAM for Ollama's default 5 minutes after the
last request, then is unloaded.

## Disk space
Ollama needs at least 4 GB to install. Models: `llama3.2:3b` 2.0 GB, `qwen3:4b-instruct` 2.5 GB. Check first:
`df -h /c`. After choosing a model, remove the other one (`ollama rm <model>`).

## Install (once)
1. Download `OllamaSetup.exe` from https://ollama.com/download (Windows). No administrator rights needed; it
   installs into your user folder (`%LOCALAPPDATA%\Programs\Ollama`, models in `%USERPROFILE%\.ollama`).
2. The installer starts the Ollama tray app. **Quit it:** right-click the llama icon in the taskbar tray -> Quit Ollama.
3. **Disable start at login:** Task Manager (Ctrl+Shift+Esc) -> Startup apps -> Ollama -> Disable.
   Check that the shortcut is gone or disabled (Git Bash):
   `ls "$APPDATA/Microsoft/Windows/Start Menu/Programs/Startup"`
   If `Ollama.lnk` is listed, delete it: `rm "$APPDATA/Microsoft/Windows/Start Menu/Programs/Startup/Ollama.lnk"`.
   Ollama updates can re-create it; check again after an update.
4. Close and reopen VS Code completely (it picks up the new PATH), then: `ollama --version`.

## Start (only when using the app)
In a separate terminal, leave this running:

    ollama serve

Always start `ollama serve` before other `ollama` commands: if no server is running, the CLI may launch the tray
app instead (then quit it from the tray as in step 2).

## Stop fully
1. Ctrl+C in the `ollama serve` terminal.
2. Check that nothing is left (Git Bash; the `//` is needed in Git Bash):

       tasklist //FI "IMAGENAME eq ollama.exe"
       tasklist //FI "IMAGENAME eq ollama app.exe"

   Both should say "No tasks are running". Otherwise: `taskkill //IM "ollama app.exe" //F` and
   `taskkill //IM ollama.exe //F`.

Free the RAM without stopping the server: `ollama ps` (shows loaded models) then `ollama stop <model>`.

## Choosing the model (measured, step 2)
`python scripts/benchmark_llm.py --models qwen3:4b-instruct llama3.2:3b` runs both on the three example evidence
files with an interim prompt and reports speed, RAM and crude wording red flags. Plain `qwen3:4b` is NOT used: that
tag is the "thinking" variant, which writes long hidden reasoning first (slow on CPU).

### Results (measured once on this laptop: Ryzen 5 5500U, 15.3 GB RAM, CPU only; Ollama default settings)
Prompt = interim fact list plus rules (585-851 prompt tokens). One run per file, so timings are indicative, not averages.
Load time (first request only): llama 4.2 s, qwen 6.7 s.

| model | evidence | total s | prompt read tok/s | output tok/s | RAM loaded | real wording violations |
|---|---|---|---|---|---|---|
| llama3.2:3b | image | 28.9 | 63 | 14.0 | 2.6 GB | "65% confident" (score as certainty) |
| llama3.2:3b | video | 19.9 | 86 | 13.9 | | "99% confident" (score as certainty) |
| llama3.2:3b | no_face | 17.0 | 95 | 15.0 | | invented a verdict: "fake-class probability of 0.91 ... predicted that the video is manipulated" |
| qwen3:4b-instruct | image | 61.9 | 27 | 7.5 | 3.2 GB | none |
| qwen3:4b-instruct | video | 49.5 | 32 | 7.5 | | none |
| qwen3:4b-instruct | no_face | 27.6 | 37 | 8.1 | | none |

How the flags were judged (each answer was read, not only counted):
- First-run flags that were **false positives** and were removed from the checker: quoting the required disclaimer
  "not ground-truth manipulated pixels" (both models); "certain types of images" ("certain" = "some"); the word
  "manipulated" inside hedged sentences ("likelihood that ... is manipulated").
- The 0.91 in llama's NO_FACE answer is the *example* number in `score_semantics`, presented as a result. The checker
  therefore never treats `score_semantics` numbers as evidence.
- qwen's "brightness was increased by 10%" is a correct paraphrase of "brightness x1.1", still flagged because
  shown numbers should match the evidence exactly.
- Not caught by the checker, found by reading: both NO_FACE answers said "video" for an image. Cause: the interim
  fact list omitted the media type for NO_FACE (fixed in the script; the step 3 digest always includes it). qwen's
  video answer tied the 3 key-frame regions to the 0-5 s interval (loosely worded). Such errors are why step 4's
  deterministic checks decide what is shown, and why the facts given to the LLM must be complete.

**Choice: `qwen3:4b-instruct`.** Reason: no real wording violation on the three examples, and in particular no
verdict on NO_FACE, while llama3.2:3b stated scores as certainty twice and invented a NO_FACE verdict. Trade-off:
about 2-2.5x slower (28-62 s per explanation, expected to vary with prompt length) and 0.6 GB more RAM. Three files
are a small sample: this is a choice between two candidates, not a quality measurement. Explanation quality in
general is **not yet measured**. `llama3.2:3b` was removed afterwards (`ollama rm`).

## Explanation timing (step 3, measured once, qwen3:4b-instruct, CPU)
`scripts/explain_evidence.py`, one run per file; the first file includes loading the model.

| version | example_image | PixVerse video |
|---|---|---|
| first prompt (all sections incl. limitations, no length cap) | 131.0 s | 115.6 s |
| limitations shown verbatim instead, <= 2 sentences / 40 words per section, 500-token cap | 80.1 s (991 prompt / 266 output tokens) | 70.7 s (1175 / 291) |

Still too slow to wait for, so the app shows the template report at once and fills in the LLM sections when they
are ready (steps 6 and 8). Whether Ollama reuses the identical system prompt between requests was not confirmed:
the reported prompt token counts did not drop.

## Uninstall (after the project)
1. Stop Ollama fully (above).
2. Optional, to see what is removed: `ollama list`.
3. Windows Settings -> Apps -> Installed apps -> Ollama -> Uninstall.
4. Delete what the uninstaller can leave behind (models, logs):

       rm -rf "$HOME/.ollama"
       rm -rf "$LOCALAPPDATA/Ollama"
       rm -rf "$LOCALAPPDATA/Programs/Ollama"

5. Check that no Startup shortcut remains (install step 3) and that no Ollama folder remains in your user PATH
   (Settings -> "Edit environment variables for your account" -> Path).
