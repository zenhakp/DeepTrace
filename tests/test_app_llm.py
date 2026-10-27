"""The Ollama client against a small fake Ollama server on 127.0.0.1 (no real LLM needed)."""
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from deeptrace.app.llm import DEFAULT_CONFIG, LLMSettings, LLMUnavailable, chat, check, loaded, unload

CHAT_OK = {"model": "qwen3:4b-instruct", "message": {"role": "assistant", "content": "  Explained.  "},
           "done": True, "prompt_eval_count": 120, "eval_count": 30, "load_duration": 2_000_000_000,
           "prompt_eval_duration": 3_000_000_000, "eval_duration": 4_000_000_000, "total_duration": 9_500_000_000}
TAGS = {"models": [{"name": "qwen3:4b-instruct"}, {"name": "llama3.2:latest"}]}
MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def server():
    """routes: {(method, path): (status, body or bytes, delay_s)}; requests: received (method, path, json body)."""
    routes, requests = {}, []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            requests.append((method, self.path, json.loads(self.rfile.read(n)) if n else None))
            status, body, delay = routes.get((method, self.path), (404, {"error": "no route"}, 0))
            time.sleep(delay)
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._reply("GET")

        def do_POST(self):
            self._reply("POST")

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield LLMSettings(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout_s=5), routes, requests
    httpd.shutdown()
    httpd.server_close()


def _code(fn, *args):
    with pytest.raises(LLMUnavailable) as e:
        fn(*args)
    return e.value.code


def test_chat_returns_text_and_timings(server):
    s, routes, requests = server
    routes[("POST", "/api/chat")] = (200, CHAT_OK, 0)
    r = chat(MSG, s)
    assert r.text == "Explained." and r.model == "qwen3:4b-instruct"
    assert (r.prompt_tokens, r.output_tokens) == (120, 30)
    assert (r.load_s, r.prompt_s, r.output_s, r.total_s) == (2.0, 3.0, 4.0, 9.5)


def test_chat_sends_model_options_and_no_streaming(server):
    s, routes, requests = server
    routes[("POST", "/api/chat")] = (200, CHAT_OK, 0)
    chat(MSG, s)
    body = requests[-1][2]
    assert body["model"] == s.model and body["messages"] == MSG and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 4096}


def test_not_running():
    with socket.socket() as sock:  # a port that nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    s = LLMSettings(base_url=f"http://127.0.0.1:{port}", timeout_s=5)
    for fn, args in ((chat, (MSG, s)), (check, (s,))):  # Windows takes ~2 s to report a refused connection
        with pytest.raises(LLMUnavailable) as e:
            fn(*args)
        assert e.value.code == "not_running" and "ollama serve" in str(e.value)


def test_model_missing_on_404(server):
    s, routes, _ = server
    routes[("POST", "/api/chat")] = (404, {"error": "model 'qwen3:4b-instruct' not found"}, 0)
    with pytest.raises(LLMUnavailable) as e:
        chat(MSG, s)
    assert e.value.code == "model_missing" and "ollama pull qwen3:4b-instruct" in str(e.value)


@pytest.mark.parametrize("status,body", [
    (500, {"error": "out of memory"}),
    (200, b"<html>not json</html>"),
    (200, {"done": True}),
    (200, {**CHAT_OK, "message": {"role": "assistant", "content": "   "}}),
])
def test_bad_response(server, status, body):
    s, routes, _ = server
    routes[("POST", "/api/chat")] = (status, body, 0)
    assert _code(chat, MSG, s) == "bad_response"


def test_timeout(server):
    s, routes, _ = server
    routes[("POST", "/api/chat")] = (200, CHAT_OK, 1.5)
    assert _code(chat, MSG, LLMSettings(base_url=s.base_url, timeout_s=0.3)) == "timeout"


def test_check_lists_models_and_accepts_latest_tag(server):
    s, routes, _ = server
    routes[("GET", "/api/tags")] = (200, TAGS, 0)
    assert check(s) == ["qwen3:4b-instruct", "llama3.2:latest"]
    check(LLMSettings(base_url=s.base_url, model="llama3.2"))  # "llama3.2" means "llama3.2:latest"


def test_check_model_missing(server):
    s, routes, _ = server
    routes[("GET", "/api/tags")] = (200, TAGS, 0)
    assert _code(check, LLMSettings(base_url=s.base_url, model="llama3.2:3b")) == "model_missing"


def test_loaded_and_unload(server):
    s, routes, requests = server
    routes[("GET", "/api/ps")] = (200, {"models": [{"name": "qwen3:4b-instruct", "size": 3_400_000_000}]}, 0)
    routes[("POST", "/api/generate")] = (200, {"done": True}, 0)
    assert loaded(s) == [{"name": "qwen3:4b-instruct", "size_bytes": 3_400_000_000}]
    unload(s)
    assert requests[-1][2] == {"model": s.model, "keep_alive": 0}


def test_settings_from_config():
    s = LLMSettings.from_config()
    assert DEFAULT_CONFIG.name == "part3_app.yaml"
    assert s.base_url.startswith("http://127.0.0.1") and s.model == "qwen3:4b-instruct" and s.seed == 42
    assert LLMSettings.from_config(model="llama3.2:3b").model == "llama3.2:3b"


def test_llm_client_does_not_import_torch():
    code = "import sys, deeptrace.app.llm; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
