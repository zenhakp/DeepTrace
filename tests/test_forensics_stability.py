import json

import pytest

from deeptrace.forensics import load_forensics_config
from deeptrace.forensics.stability import band, components, stability, weight_sensitivity

W = {"prediction": 0.4, "confidence": 0.3, "localization": 0.3}
B = {"high": 0.8, "medium": 0.6}


def _row(p, base_label, corr=1.0, mode="pipeline"):
    if p is None:
        return {"mode": mode, "fake_probability": None, "prediction": None, "prediction_flipped": None,
                "no_face": True, "localization": None}
    label = "MANIPULATED" if p >= 0.5 else "REAL"
    return {"mode": mode, "fake_probability": p, "prediction": label, "prediction_flipped": label != base_label,
            "no_face": False, "localization": {"map_correlation": corr}}


def _report(p0, ps, corrs=None, mode="pipeline"):
    base = "MANIPULATED" if p0 >= 0.5 else "REAL"
    corrs = corrs or [1.0] * len(ps)
    return {"status": "OK", "baseline": {"prediction": base, "fake_probability": p0},
            "perturbations": [_row(p, base, c, mode) for p, c in zip(ps, corrs)]}


def test_unchanged_evidence_scores_one_and_has_no_driver():
    st = stability(_report(0.05, [0.05, 0.05]), weights=W, bands=B)
    json.dumps(st)
    e = st["evidence_stability"]
    assert e["score"] == 1.0 and e["band"] == "HIGH" and e["driver"] is None
    assert "not a probability" in e["summary"] and "not a probability" in st["note"]


def test_the_same_change_costs_more_near_the_decision_boundary():
    far = components([_row(0.15, "REAL")], baseline_p=0.05)["confidence_stability"]["value"]      # margin 0.45
    near = components([_row(0.75, "MANIPULATED")], baseline_p=0.65)["confidence_stability"]["value"]  # margin 0.15
    assert far == pytest.approx(1 - 0.10 / 0.45, abs=1e-4) and near == pytest.approx(1 - 0.10 / 0.15, abs=1e-4)
    assert near < far


def test_a_flip_uses_up_the_whole_margin_and_the_floor_handles_p_half():
    c = components([_row(0.55, "REAL")], baseline_p=0.45)
    assert c["prediction_stability"]["value"] == 0 and c["confidence_stability"]["value"] == 0
    assert components([_row(0.505, "MANIPULATED")], 0.5)["confidence_stability"]["baseline_margin"] == 0.01


def test_no_face_runs_count_as_lost_evidence():
    c = components([_row(None, "REAL"), _row(0.05, "REAL")], baseline_p=0.05)
    ps = c["prediction_stability"]
    assert ps["value"] == 0.5 and ps["no_face"] == 1 and ps["flips"] == 0
    assert c["confidence_stability"]["value"] == 0.5 and c["localization_stability"]["value"] == 0.5
    assert c["localization_stability"]["compared_runs"] == 1


def test_negative_correlation_is_clipped_and_drives_the_score():
    st = stability(_report(0.02, [0.02, 0.02], corrs=[-0.4, 0.2]), weights=W, bands=B)
    assert st["localization_stability"]["value"] == pytest.approx(0.1)
    e = st["evidence_stability"]
    assert e["driver"] == "localization" and e["score"] == pytest.approx(0.4 + 0.3 + 0.03) and e["band"] == "MEDIUM"


def test_band_edges():
    assert [band(s, B) for s in (0.8, 0.7999, 0.6, 0.5999)] == ["HIGH", "MEDIUM", "MEDIUM", "LOW"]


def test_only_the_chosen_mode_feeds_the_score():
    rep = _report(0.05, [0.05])
    rep["perturbations"].append(_row(0.9, "REAL", 0.0, mode="fixed_crop"))
    st = stability(rep, mode="pipeline", weights=W, bands=B)
    assert st["evidence_stability"]["score"] == 1.0
    assert st["diagnostics"]["fixed_crop"]["evidence_stability"]["score"] == 0.0
    with pytest.raises(ValueError, match="no 'video' runs"):
        stability(rep, mode="video", weights=W, bands=B)


def test_invalid_settings_fail_loudly():
    with pytest.raises(ValueError, match="sum to 1"):
        stability(_report(0.05, [0.05]), weights={"prediction": 0.5, "confidence": 0.5, "localization": 0.5})
    with pytest.raises(ValueError, match="exactly"):
        stability(_report(0.05, [0.05]), weights={"prediction": 1.0})
    with pytest.raises(ValueError, match="bands"):
        stability(_report(0.05, [0.05]), weights=W, bands={"high": 0.5, "medium": 0.7})


def test_no_face_baseline_is_not_applicable():
    st = stability({"status": "NOT_APPLICABLE", "reason": "No face", "baseline": {}, "perturbations": []})
    assert st["status"] == "NOT_APPLICABLE" and st["evidence_stability"] is None


def test_config_matches_the_documented_defaults():
    cfg = load_forensics_config()["stability"]
    assert cfg["mode"] == "pipeline" and cfg["weights"] == W and cfg["bands"] == B and cfg["margin_floor"] == 0.01
    assert cfg["cap_band"] == "MEDIUM" and len(cfg["sensitivity_weights"]) >= 2


def test_a_flip_caps_a_high_score_at_medium_without_changing_the_score():
    ps = [0.05] * 11 + [0.6]                              # one flip in 12 runs, otherwise unchanged
    e = stability(_report(0.05, ps), weights=W, bands=B)["evidence_stability"]
    assert e["band_uncapped"] == "HIGH" and e["band"] == "MEDIUM" and e["band_capped"] is True
    assert "1 prediction flip(s)" in e["cap_reason"] and "capped from HIGH" in e["summary"]
    uncapped = stability(_report(0.05, ps), weights=W, bands=B, cap_band=None)["evidence_stability"]
    assert uncapped["score"] == e["score"] and uncapped["band"] == "HIGH" and uncapped["band_capped"] is False


def test_no_face_also_caps_and_low_is_never_raised():
    e = stability(_report(0.05, [0.05] * 11 + [None]), weights=W, bands=B)["evidence_stability"]
    assert e["band"] == "MEDIUM" and e["band_capped"] and "1 lost face(s)" in e["cap_reason"]
    low = stability(_report(0.05, [0.9, 0.9, 0.05]), weights=W, bands=B)["evidence_stability"]
    assert low["band"] == "LOW" and low["band_capped"] is False and low["cap_reason"] is None


def test_summary_says_what_the_cost_means():
    e = stability(_report(0.05, [0.05, 0.05], corrs=[0.5, 0.5]), weights=W, bands=B)["evidence_stability"]
    assert "Largest cost: localization stability 0.50 (costs 0.15 of 1.00 points)." in e["summary"]


def test_weight_sensitivity_counts_band_changes():
    near_edge = {"image": "a.jpg", **stability(_report(0.05, [0.05, 0.05], corrs=[0.4, 0.4]), weights=W, bands=B)}
    stable = {"image": "b.jpg", **stability(_report(0.05, [0.05, 0.05]), weights=W, bands=B)}
    assert near_edge["evidence_stability"]["band"] == "HIGH"          # 0.4 + 0.3 + 0.12 = 0.82
    rows = weight_sensitivity([near_edge, stable], {"loc_heavy": {"prediction": 0.2, "confidence": 0.2,
                                                                  "localization": 0.6}}, B)
    assert rows[0]["changed"] == 1 and rows[0]["changes"][0]["image"] == "a.jpg"
    assert rows[0]["changes"][0]["alt_band"] == "MEDIUM" and rows[0]["images"] == 2
