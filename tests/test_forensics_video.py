import json
from collections import OrderedDict

import numpy as np
import pytest
import torch

from deeptrace.forensics import load_forensics_config
from deeptrace.forensics.video import (analyze_video, iter_frames, key_frames, suspicious_intervals,
                                       temporal_stability, video_info)
from deeptrace.models.detector import Detector
from deeptrace.models.inference import DeepTraceDetector

FACE = (40.0, 40.0, 60.0, 60.0, 0.9)
PARAMS = {"crop_margin": 1.3, "jpeg_quality": 95}


def _f(i, p, box=(0, 0, 10, 10)):
    """A frame record as analyze_video produces it (p=None: no face)."""
    return {"frame": i * 12, "time_s": i * 0.5, "fake_probability": p,
            "prediction": None if p is None else ("MANIPULATED" if p >= 0.5 else "REAL"),
            "crop_box": None if p is None else list(box)}


def test_intervals_merge_small_gaps_and_report_isolated_frames():
    frames = [_f(i, p) for i, p in enumerate([0.1, 0.8, 0.9, 0.3, 0.7, 0.2, 0.2, 0.95, 0.1])]
    intervals, isolated = suspicious_intervals(frames, threshold=0.5, min_frames=2, merge_gap=1)
    assert len(intervals) == 1 and len(isolated) == 1
    iv = intervals[0]
    assert (iv["start_s"], iv["end_s"], iv["sampled_frames"], iv["suspicious_frames"]) == (0.5, 2.0, 4, 3)
    assert iv["max_fake_probability"] == 0.9 and iv["peak_time_s"] == 1.0
    assert isolated[0]["start_s"] == isolated[0]["end_s"] == 3.5


def test_no_face_frames_are_bridged_only_within_the_gap():
    frames = [_f(0, 0.8), _f(1, None), _f(2, 0.8), _f(3, None), _f(4, None), _f(5, 0.8)]
    intervals, isolated = suspicious_intervals(frames, 0.5, 2, 1)
    assert [(i["start_s"], i["end_s"]) for i in intervals] == [(0.0, 1.0)] and len(isolated) == 1


def test_temporal_stability_counts_steady_pairs_and_face_jumps():
    frames = [_f(0, 0.1), _f(1, 0.15), _f(2, None), _f(3, 0.7, box=(50, 50, 60, 60)), _f(4, 0.72, box=(50, 50, 60, 60))]
    ts = temporal_stability(frames, "REAL", max_jump=0.2, face_jump_iou=0.3)
    assert ts["pairs"] == 3 and ts["steady_pairs"] == 2 and ts["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert ts["label_flips"] == 1 and ts["label_agreement"] == 0.5
    assert [j["to_frame"] for j in ts["face_jumps"]] == [36]
    assert temporal_stability([_f(0, 0.1), _f(1, None)], "REAL") is None


def test_key_frames_are_the_most_suspicious_plus_the_median():
    frames = [_f(i, p) for i, p in enumerate([0.1, 0.9, None, 0.3, 0.8, 0.2])]
    assert key_frames(frames, top_suspicious=2, include_median=True) == [1, 3, 4]
    assert key_frames([_f(0, None)], 2, True) == []


def _write_video(path, brightness, fps=10.0, size=(160, 120)):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    rng = np.random.default_rng(0)
    for b in brightness:
        frame = np.clip(rng.integers(90, 160, (size[1], size[0], 3)) * b, 0, 255).astype(np.uint8)
        w.write(frame)
    w.release()
    return path


def test_frames_are_sampled_on_a_fixed_grid(tmp_path):
    path = _write_video(tmp_path / "v.avi", [1.0] * 20, fps=10.0)
    assert video_info(path)["fps"] == pytest.approx(10.0) and video_info(path)["frame_count"] == 20
    got = [(i, t) for i, t, _ in iter_frames(path, sample_fps=2.0, max_frames=100)]
    assert got == [(0, 0.0), (5, 0.5), (10, 1.0), (15, 1.5)]
    assert [i for i, _, _ in iter_frames(path, 2.0, 2)] == [0, 5]
    assert [i for i, _, _ in iter_frames(path, 2.0, 100, only=[7])] == [7]


def test_missing_and_unreadable_videos_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        video_info(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="Unable to read"):
        video_info(bad)


def _det():
    """MANIPULATED logit grows with red brightness; the face detector fails on dark frames."""
    m = torch.nn.Sequential(OrderedDict(feat=torch.nn.Conv2d(3, 1, 1, bias=False), pool=torch.nn.AdaptiveAvgPool2d(1),
                                        flat=torch.nn.Flatten(), fc=torch.nn.Linear(1, 2)))
    with torch.no_grad():
        m.feat.weight.copy_(torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1))   # frames are BGR on disk: red = index 2
        m.fc.weight.copy_(torch.tensor([[0.0], [10.0]]))
        m.fc.bias.copy_(torch.tensor([0.0, -2.0]))
    return DeepTraceDetector(Detector(m.eval(), "cpu"), {"epoch": 3, "image_size": 64},
                             lambda rgb: FACE if rgb.mean() > 40 else None, PARAMS, checkpoint="x/best.pt",
                             min_crop_side=32, model_version="test")


def _cfg():
    cfg = load_forensics_config()
    cfg["attribution"]["target_layer"] = "feat"
    cfg["robustness"]["perturbations"] = {"brightness": [0.9]}
    return cfg


def test_analyze_video_end_to_end_records_no_face_frames(tmp_path):
    # 2 fps sampling of a 10 fps clip: bright, bright, dark (no face), dark, bright, bright
    path = _write_video(tmp_path / "v.avi", [1.5] * 10 + [0.1] * 10 + [1.5] * 10, fps=10.0)
    rep = analyze_video(_det(), path, _cfg())
    json.dumps(rep)
    assert rep["status"] == "OK" and rep["sampling"]["sampled_frames"] == 6
    assert rep["usable_face_frames"] == 4 and rep["usable_face_fraction"] == pytest.approx(4 / 6, abs=1e-4)
    no_face = [o for o in rep["observations"] if o["kind"] == "no_face_frames"]
    assert len(no_face) == 1 and no_face[0]["frames"] == 2 and (no_face[0]["start_s"], no_face[0]["end_s"]) == (1.0, 1.5)
    assert rep["prediction"] == "MANIPULATED" and rep["aggregation"]["manipulated_frame_fraction"] == 1.0
    # the 2-frame no-face gap is wider than merge_gap_frames=1, so it splits the suspicious run in two
    assert [(i["start_s"], i["end_s"]) for i in rep["suspicious_intervals"]] == [(0.0, 0.5), (2.0, 2.5)]
    assert rep["temporal_stability"]["pairs"] == 3 and len(rep["key_frames"]) == 3
    assert rep["evidence_stability"]["method"].startswith("median") and "downward" in rep["evidence_stability"]["bias_note"]
    assert rep["model"]["version"] == "test" and rep["limitations"]


def test_video_without_any_face_is_no_face(tmp_path):
    path = _write_video(tmp_path / "dark.avi", [0.1] * 10, fps=10.0)
    rep = analyze_video(_det(), path, _cfg())
    assert rep["status"] == "NO_FACE" and rep["prediction"] is None and rep["usable_face_fraction"] == 0
    assert rep["evidence_stability"] is None and rep["temporal_stability"] is None
    assert [w["code"] for w in rep["warnings"]] == ["low_face_coverage"]
    assert rep["observations"][0]["kind"] == "no_face_frames"
