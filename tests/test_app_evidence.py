import copy
import json
import subprocess
import sys

import pytest

from deeptrace.app import evidence as ev_mod
from deeptrace.app.evidence import REPO_ROOT, EvidenceError, check_evidence, load_evidence, parse_evidence

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def _doc(name):
    return json.loads((EXAMPLES / f"example_{name}.json").read_text(encoding="utf-8"))


def _code(doc):
    with pytest.raises(EvidenceError) as e:
        check_evidence(doc)
    return e.value


@pytest.mark.parametrize("name", ["image", "video", "no_face"])
def test_examples_load(name):
    ev = load_evidence(EXAMPLES / f"example_{name}.json")
    assert ev.data["schema_version"] == "1.0"
    assert ev.has_prediction == (name != "no_face")
    assert ev.is_video == (name == "video")
    assert ev.source.name == f"example_{name}.json"


def test_overlay_files():
    assert load_evidence(EXAMPLES / "example_image.json").overlay_files() == {"image": "example_image_overlay.png"}
    assert load_evidence(EXAMPLES / "example_video.json").overlay_files() == {
        "24": "example_video_frame24_overlay.png", "48": "example_video_frame48_overlay.png",
        "120": "example_video_frame120_overlay.png"}


@pytest.mark.parametrize("bad", ["../secret.png", "sub/x.png", "..\\x.png", "C:\\x.png", "/etc/x", "..", "", 5, None])
def test_overlay_files_never_leave_the_job_folder(bad):
    d = _doc("image")
    d["artifacts"] = {"overlay": bad}
    assert check_evidence(d).overlay_files() == {}


def test_unknown_extra_fields_are_allowed_and_kept():
    d = _doc("image")
    d["new_top_level_field"] = {"anything": 1}
    d["evidence_stability"]["new_nested_field"] = [1, 2]
    ev = check_evidence(d)
    assert ev.data["new_top_level_field"] == {"anything": 1}


@pytest.mark.parametrize("version", ["2.0", "1.1", 1.0, None])
def test_unsupported_schema_version(version):
    d = _doc("image")
    d["schema_version"] = version
    assert _code(d).code == "unsupported_schema_version"


def test_missing_schema_version():
    d = _doc("image")
    del d["schema_version"]
    assert _code(d).code == "unsupported_schema_version"


def test_missing_required_key():
    d = _doc("video")
    del d["limitations"]
    e = _code(d)
    assert e.code == "schema_violation" and any("limitations" in m for m in e.errors)


@pytest.mark.parametrize("field,value,where", [
    ("confidence", 1.5, "$.confidence"),
    ("prediction", "FAKE", "$.prediction"),
    ("status", "ERROR", "$.status"),
    ("suspicious_regions", "none", "$.suspicious_regions"),
])
def test_schema_violations_name_the_field(field, value, where):
    d = _doc("image")
    d[field] = value
    e = _code(d)
    assert e.code == "schema_violation" and any(m.startswith(where) for m in e.errors)


def test_nested_violation_path():
    d = _doc("image")
    d["suspicious_regions"][1]["rank"] = 0
    assert any(m.startswith("$.suspicious_regions[1].rank") for m in _code(d).errors)


@pytest.mark.parametrize("field,value", [("prediction", "REAL"), ("confidence", 0.9), ("fake_probability", 0.1),
                                         ("real_probability", 0.9), ("temporal_stability", None)])
def test_no_face_must_not_carry_a_verdict(field, value):
    d = _doc("no_face")
    d[field] = value if value is not None else _doc("video")[field]  # a schema-valid object, so only check 4 fires
    e = _code(d)
    assert e.code == "inconsistent" and any(m.startswith(f"$.{field}") for m in e.errors)


def test_no_face_with_evidence_stability_is_inconsistent():
    d = _doc("no_face")
    d["evidence_stability"] = copy.deepcopy(_doc("image")["evidence_stability"])
    assert _code(d).code == "inconsistent"


def test_no_face_with_regions_is_inconsistent():
    d = _doc("no_face")
    d["suspicious_regions"] = _doc("image")["suspicious_regions"]
    assert _code(d).code == "inconsistent"


@pytest.mark.parametrize("field", ["prediction", "confidence", "fake_probability", "real_probability"])
def test_ok_needs_a_verdict(field):
    d = _doc("image")
    d[field] = None
    assert _code(d).code == "inconsistent"


@pytest.mark.parametrize("label,p", [("REAL", 0.9), ("MANIPULATED", 0.2)])
def test_prediction_contradicting_probability(label, p):
    d = _doc("image")
    d["prediction"], d["fake_probability"] = label, p
    assert _code(d).code == "inconsistent"


def test_probability_exactly_half_is_accepted_for_either_label():
    for label in ("REAL", "MANIPULATED"):
        d = _doc("image")
        d["prediction"], d["fake_probability"] = label, 0.5
        check_evidence(d)


def test_image_with_frames_and_video_without_frames():
    d = _doc("image")
    d["frames"] = []
    assert _code(d).code == "inconsistent"
    v = _doc("video")
    v["frames"] = None
    assert _code(v).code == "inconsistent"


@pytest.mark.parametrize("raw", [b"{not json", b"", b"\xff\xfe\x00", b'{"confidence": NaN}', "Infinity"])
def test_invalid_json(raw):
    with pytest.raises(EvidenceError) as e:
        parse_evidence(raw)
    assert e.value.code == "invalid_json"


@pytest.mark.parametrize("raw", [b"[]", b'"text"', b"42"])
def test_top_level_must_be_an_object(raw):
    with pytest.raises(EvidenceError) as e:
        parse_evidence(raw)
    assert e.value.code == "schema_violation"


def test_byte_order_mark_is_accepted():
    raw = (EXAMPLES / "example_image.json").read_bytes()
    assert parse_evidence(b"\xef\xbb\xbf" + raw).has_prediction
    assert parse_evidence("\ufeff" + raw.decode("utf-8")).has_prediction


def test_not_found(tmp_path):
    for p in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(EvidenceError) as e:
            load_evidence(p)
        assert e.value.code == "not_found"


def test_too_large(tmp_path, monkeypatch):
    f = tmp_path / "e.json"
    f.write_bytes((EXAMPLES / "example_image.json").read_bytes())
    monkeypatch.setattr(ev_mod, "MAX_BYTES", 100)
    with pytest.raises(EvidenceError) as e:
        load_evidence(f)
    assert e.value.code == "too_large"


def test_error_message_lists_code_and_errors():
    d = _doc("image")
    d["confidence"] = 2
    e = _code(d)
    assert str(e).startswith("schema_violation: $.confidence")


def test_loading_evidence_does_not_import_torch():
    code = "import sys, deeptrace.app.evidence; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
