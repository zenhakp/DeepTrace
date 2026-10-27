"""run_analysis() with fake subprocess commands (no detector or checkpoint needed)."""
import sys
from pathlib import Path

import pytest

from deeptrace.app.evidence import REPO_ROOT, EvidenceError
from deeptrace.app.pipeline import (ANALYZE_SCRIPT, PipelineError, analyze_command, checkpoint_path, media_type_of,
                                    run_analysis)

EXAMPLE = REPO_ROOT / "docs" / "part2_evidence_examples" / "example_image.json"


def fake(code: str):
    """A command running `code` with sys.argv = [input, out_dir]."""
    return lambda inp, out, run: [sys.executable, "-c", code, str(inp), str(out)]


WRITE_EXAMPLE = ("import sys, shutil, pathlib; i, o = map(pathlib.Path, sys.argv[1:3]); "
                 f"shutil.copy(r'{EXAMPLE}', o / f'{{i.stem}}_evidence.json')")


@pytest.fixture
def upload(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"not really a jpeg")
    return p


def test_success_loads_and_checks_the_evidence(upload, tmp_path):
    ev = run_analysis(upload, tmp_path, command=fake(WRITE_EXAMPLE))
    assert ev.data["prediction"] == "MANIPULATED" and ev.source == tmp_path / "photo_evidence.json"


def test_failure_keeps_the_last_error_lines(upload, tmp_path):
    with pytest.raises(PipelineError) as e:
        run_analysis(upload, tmp_path, command=fake("import sys; print('step 1', file=sys.stderr); "
                                                    "raise SystemExit('no face model')"))
    assert e.value.code == "analysis_failed" and "no face model" in str(e.value) and "step 1" in str(e.value)


def test_timeout(upload, tmp_path):
    with pytest.raises(PipelineError) as e:
        run_analysis(upload, tmp_path, timeout_s=1, command=fake("import time; time.sleep(10)"))
    assert e.value.code == "analysis_timeout"


def test_no_evidence_written(upload, tmp_path):
    with pytest.raises(PipelineError) as e:
        run_analysis(upload, tmp_path, command=fake("pass"))
    assert e.value.code == "evidence_missing"


def test_invalid_evidence_is_refused(upload, tmp_path):
    code = "import sys, pathlib; pathlib.Path(sys.argv[2], 'photo_evidence.json').write_text('{bad')"
    with pytest.raises(EvidenceError) as e:
        run_analysis(upload, tmp_path, command=fake(code))
    assert e.value.code == "invalid_json"


def test_missing_checkpoint_is_reported_before_starting(upload, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPTRACE_ROOT", str(tmp_path / "empty_root"))
    with pytest.raises(PipelineError) as e:
        run_analysis(upload, tmp_path)
    assert e.value.code == "checkpoint_missing" and "DEEPTRACE_ROOT" in str(e.value)
    assert checkpoint_path("mixed") == (tmp_path / "empty_root" / "checkpoints" / "mixed" / "best.pt").resolve()


def test_real_command_runs_part2_analyze_script(tmp_path):
    cmd = analyze_command(Path("in.jpg"), tmp_path, "mixed")
    assert cmd[0] == sys.executable and cmd[1] == str(ANALYZE_SCRIPT) and ANALYZE_SCRIPT.is_file()
    assert cmd[2:] == ["--input", "in.jpg", "--out-dir", str(tmp_path), "--run-name", "mixed"]


@pytest.mark.parametrize("name,kind", [("a.JPG", "image"), ("b.webp", "image"), ("c.mp4", "video"),
                                       ("d.MOV", "video"), ("e.txt", None), ("noext", None), ("f.jpg.exe", None)])
def test_media_type_of(name, kind):
    assert media_type_of(name) == kind
