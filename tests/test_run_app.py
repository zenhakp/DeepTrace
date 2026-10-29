"""scripts/run_app.py helpers (the servers themselves are covered by the API tests and the live checks)."""
import importlib.util
import socket
import subprocess
import sys
import time

from deeptrace.app.evidence import REPO_ROOT

spec = importlib.util.spec_from_file_location("run_app", REPO_ROOT / "scripts" / "run_app.py")
run_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_app)


def test_backend_runs_the_api_on_localhost_with_this_python():
    assert run_app.backend_command() == [sys.executable, "-m", "uvicorn", "deeptrace.app.api:app",
                                         "--host", "127.0.0.1", "--port", "8000"]


def test_frontend_runs_next_directly_on_localhost():
    cmd = run_app.frontend_command("node")
    assert cmd[0] == "node" and cmd[1] == str(REPO_ROOT / "frontend" / "node_modules" / "next" / "dist" / "bin" / "next")
    assert cmd[2:] == ["start", "--hostname", "127.0.0.1", "--port", "3000"]


def test_port_in_use():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert run_app.port_in_use(port)
    assert not run_app.port_in_use(port)


def test_stop_tree_stops_a_process_and_its_children():
    code = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code])
    time.sleep(1.0)
    run_app.stop_tree(proc)
    assert proc.poll() is not None
    run_app.stop_tree(proc)  # stopping twice is harmless


def test_wait_for_gives_up_when_the_process_exits():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    started = time.time()
    assert run_app.wait_for("http://127.0.0.1:9/", 30, proc) is False
    assert time.time() - started < 5
