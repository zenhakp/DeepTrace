"""Start the DeepTrace app locally: FastAPI backend (127.0.0.1:8000) and Next.js frontend (127.0.0.1:3000).

  .venv/Scripts/python.exe scripts/run_app.py              backend + frontend; uses Ollama if it is running
  .venv/Scripts/python.exe scripts/run_app.py --with-llm   also starts `ollama serve`, and stops it again on exit
  .venv/Scripts/python.exe scripts/run_app.py --build      rebuild the frontend first (after changing its code)
  .venv/Scripts/python.exe scripts/run_app.py --stop       stop leftover DeepTrace servers on ports 8000 / 3000
Open http://127.0.0.1:3000. Ctrl+C stops everything this script started (the whole process tree, so no server is
left running on Windows). Server output goes to ~/Downloads/deeptrace_outputs/app_logs/.
Needs (once): the project venv, Node.js, `npm ci` in frontend/, and DEEPTRACE_ROOT pointing at the checkpoint
folder (see README). Without Ollama the app still works: reports are shown as evidence text.
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
FRONTEND = REPO / "frontend"
LOG_DIR = Path.home() / "Downloads" / "deeptrace_outputs" / "app_logs"
HOST, API_PORT, WEB_PORT, OLLAMA_PORT = "127.0.0.1", 8000, 3000, 11434
WINDOWS = os.name == "nt"


def port_in_use(port: int, host: str = HOST) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def backend_command() -> List[str]:
    return [sys.executable, "-m", "uvicorn", "deeptrace.app.api:app", "--host", HOST, "--port", str(API_PORT)]


def frontend_command(node: str) -> List[str]:
    """`next start` run directly by node (not through npm/npx), so stopping it stops the server itself."""
    return [node, str(FRONTEND / "node_modules" / "next" / "dist" / "bin" / "next"), "start",
            "--hostname", HOST, "--port", str(WEB_PORT)]


def wait_for(url: str, timeout_s: float, proc: Optional[subprocess.Popen] = None) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False  # it exited: see its log
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def stop_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if WINDOWS:  # terminate() would stop only the top process; /T also stops its children
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


def stop_leftovers() -> None:
    """Stop python/node processes listening on 8000 / 3000 (only those: anything else is reported, not stopped)."""
    if not WINDOWS:
        sys.exit("--stop is for Windows; elsewhere stop the servers with Ctrl+C or `kill`")
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    pids = {line.split()[-1] for line in out.splitlines() if "LISTENING" in line
            and any(line.split()[1].endswith(f":{p}") for p in (API_PORT, WEB_PORT))}
    if not pids:
        print("Nothing is listening on ports 8000 or 3000.")
    for pid in sorted(pids):
        info = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                              text=True).stdout.strip()
        image = info.split(",")[0].strip('"').lower() if info else "?"
        if image in ("python.exe", "node.exe"):
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
            print(f"Stopped {image} (PID {pid}).")
        else:
            print(f"Not stopping PID {pid} ({image}): it is not a DeepTrace server. Close it yourself if needed.")


def preflight(build: bool) -> str:
    node = shutil.which("node")
    if node is None:
        sys.exit("Node.js was not found on PATH. Install Node.js 20.9 or newer (README, Part 3).")
    if not (FRONTEND / "node_modules" / "next").is_dir():
        sys.exit("The frontend packages are not installed. Run once:  cd frontend  then  npm ci")
    for port, name in ((API_PORT, "backend"), (WEB_PORT, "frontend")):
        if port_in_use(port):
            sys.exit(f"Port {port} ({name}) is already in use; is DeepTrace still running? "
                     f"Stop it with: {Path(sys.executable).name} scripts/run_app.py --stop")
    if build or not (FRONTEND / ".next" / "BUILD_ID").is_file():
        print("Building the frontend (once, about 30 s)...")
        npm = shutil.which("npm") or "npm"
        if subprocess.run([npm, "run", "build"], cwd=FRONTEND).returncode != 0:
            sys.exit("The frontend build failed; see the output above.")
    sys.path.insert(0, str(REPO / "src"))
    from deeptrace.app.pipeline import checkpoint_path  # no torch: the app package never imports it
    ckpt = checkpoint_path("mixed")
    if not ckpt.is_file():
        print(f"WARNING: detector checkpoint not found at {ckpt}. Uploads will fail with checkpoint_missing until "
              f"DEEPTRACE_ROOT points at the folder that holds checkpoints/mixed/best.pt.")
    return node


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-llm", action="store_true", help="start `ollama serve` too (and stop it on exit)")
    p.add_argument("--build", action="store_true", help="rebuild the frontend before starting")
    p.add_argument("--stop", action="store_true", help="stop leftover DeepTrace servers and exit")
    a = p.parse_args()
    if a.stop:
        stop_leftovers()
        return
    node = preflight(a.build)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started: List[subprocess.Popen] = []

    def launch(cmd: List[str], log_name: str, cwd: Path) -> subprocess.Popen:
        log = open(LOG_DIR / log_name, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        started.append(proc)
        return proc

    try:
        if port_in_use(OLLAMA_PORT):
            print("Explanation model: Ollama is already running (it is left running on exit).")
        elif a.with_llm:
            ollama = shutil.which("ollama")
            if ollama is None:
                print("Explanation model: `ollama` not found; continuing without it (docs/part3_llm_setup.md).")
            else:
                proc = launch([ollama, "serve"], "ollama.log", REPO)
                ok = wait_for(f"http://{HOST}:{OLLAMA_PORT}/api/tags", 30, proc)
                print("Explanation model: started `ollama serve`." if ok else
                      "Explanation model: `ollama serve` did not start; see ollama.log. Continuing without it.")
        else:
            print("Explanation model: not running. Reports are shown as evidence text "
                  "(add --with-llm to start Ollama).")

        backend = launch(backend_command(), "backend.log", REPO)
        # /api/jobs, not /api/health: health also asks Ollama, which takes about 2 s to refuse on Windows
        if not wait_for(f"http://{HOST}:{API_PORT}/api/jobs", 60, backend):
            sys.exit(f"The backend did not start; see {LOG_DIR / 'backend.log'}")
        frontend = launch(frontend_command(node), "frontend.log", FRONTEND)
        if not wait_for(f"http://{HOST}:{WEB_PORT}/", 60, frontend):
            sys.exit(f"The frontend did not start; see {LOG_DIR / 'frontend.log'}")
        print(f"\nDeepTrace is running: http://{HOST}:{WEB_PORT}\nLogs: {LOG_DIR}\nPress Ctrl+C to stop.\n")
        while all(proc.poll() is None for proc in started):
            time.sleep(1)
        print("A server stopped unexpectedly; see the logs. Stopping the others.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for proc in reversed(started):
            stop_tree(proc)
        if started:
            print("Stopped everything this script started.")


if __name__ == "__main__":
    main()
