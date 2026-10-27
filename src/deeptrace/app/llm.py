"""Client for a local LLM served by Ollama (default http://127.0.0.1:11434). Standard library only.

The LLM only turns evidence it is given into prose; it is never a detector. Every failure raises LLMUnavailable with
a stable code, so the app can fall back to the deterministic report instead of crashing:
  not_running   Ollama is not reachable (start it with `ollama serve`)
  model_missing the configured model is not installed (`ollama pull <model>`)
  timeout       no answer within llm.timeout_s
  bad_response  an HTTP error or a reply without text
Models stay loaded for Ollama's default 5 minutes after the last request; unload() frees the RAM at once.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deeptrace.app.evidence import REPO_ROOT
from deeptrace.common.config import load_config

DEFAULT_CONFIG = REPO_ROOT / "configs" / "part3_app.yaml"
START_HINT = "Start it in a separate terminal with: ollama serve"
STATUS_TIMEOUT_S = 5.0


class LLMUnavailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b-instruct"
    temperature: float = 0.0
    num_ctx: int = 4096
    timeout_s: float = 300.0
    seed: int = 42

    @classmethod
    def from_config(cls, path=None, **overrides) -> "LLMSettings":
        cfg = load_config(path or DEFAULT_CONFIG)
        return cls(**{**cfg["llm"], "seed": cfg["run"]["seed"], **overrides})


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    load_s: float      # loading the model into RAM (about 0 when it was already loaded)
    prompt_s: float    # reading the prompt
    output_s: float    # writing the answer
    total_s: float


def _full_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def _request(s: LLMSettings, method: str, path: str, body: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(s.base_url.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    wait = timeout if timeout is not None else s.timeout_s
    try:
        with urllib.request.urlopen(req, timeout=wait) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", "")
        except ValueError:
            detail = ""
        if e.code == 404:
            raise LLMUnavailable("model_missing", f"{detail or 'model not found'}. Install it with: "
                                                  f"ollama pull {s.model}") from None
        raise LLMUnavailable("bad_response", f"HTTP {e.code} from Ollama: {detail or e.reason}") from None
    except TimeoutError:
        raise LLMUnavailable("timeout", f"no answer from Ollama within {wait:g} s") from None
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise LLMUnavailable("timeout", f"no answer from Ollama within {wait:g} s") from None
        raise LLMUnavailable("not_running", f"cannot reach Ollama at {s.base_url} ({e.reason}). {START_HINT}") from None
    except ConnectionError as e:
        raise LLMUnavailable("not_running", f"connection to Ollama lost ({e}). {START_HINT}") from None
    except ValueError:
        raise LLMUnavailable("bad_response", "Ollama replied with something that is not JSON") from None


def check(s: LLMSettings) -> List[str]:
    """Installed model names; raises not_running, or model_missing if s.model is not among them."""
    names = [m.get("name", "") for m in _request(s, "GET", "/api/tags", timeout=STATUS_TIMEOUT_S).get("models", [])]
    if _full_name(s.model) not in {_full_name(n) for n in names}:
        raise LLMUnavailable("model_missing", f"{s.model} is not installed (installed: {names or 'none'}). "
                                              f"Install it with: ollama pull {s.model}")
    return names


def chat(messages: List[Dict[str, str]], s: LLMSettings) -> LLMReply:
    """One non-streaming chat request. messages: [{"role": "system" | "user" | "assistant", "content": ...}]."""
    body = {"model": s.model, "messages": messages, "stream": False,
            "options": {"temperature": s.temperature, "seed": s.seed, "num_ctx": s.num_ctx}}
    r = _request(s, "POST", "/api/chat", body)
    text = (r.get("message") or {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise LLMUnavailable("bad_response", "Ollama returned no text")
    ns = lambda k: r.get(k, 0) / 1e9  # Ollama reports durations in nanoseconds
    return LLMReply(text.strip(), r.get("model", s.model), r.get("prompt_eval_count", 0), r.get("eval_count", 0),
                    ns("load_duration"), ns("prompt_eval_duration"), ns("eval_duration"), ns("total_duration"))


def loaded(s: LLMSettings) -> List[Dict[str, Any]]:
    """Models currently in RAM, as {"name", "size_bytes"} (the same as `ollama ps`)."""
    return [{"name": m.get("name"), "size_bytes": m.get("size")}
            for m in _request(s, "GET", "/api/ps", timeout=STATUS_TIMEOUT_S).get("models", [])]


def unload(s: LLMSettings) -> None:
    """Free the model's RAM now instead of after the 5-minute default (the same as `ollama stop <model>`)."""
    _request(s, "POST", "/api/generate", {"model": s.model, "keep_alive": 0}, timeout=STATUS_TIMEOUT_S * 6)
