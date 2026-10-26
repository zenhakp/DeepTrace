"""Video analysis: sampled frames -> detector + Grad-CAM -> aggregation, temporal stability, suspicious intervals,
and robustness + Evidence Stability Score on a few key frames.

  python scripts/analyze_video.py --video "path/to/clip.mp4"
Writes <stem>_video.json and key-frame overlays into --out-dir (default ~/Downloads/deeptrace_outputs/video;
must be outside the repository). Quote paths with spaces. A test clip only tests the pipeline; it is not evidence
of detection accuracy.
"""
import argparse
import json
import time
from pathlib import Path

from deeptrace.forensics import DEFAULT_OUT_DIR, load_forensics_config, output_dir
from deeptrace.forensics.attribution import explain_image
from deeptrace.forensics.localization import localize
from deeptrace.forensics.overlay import render_overlay
from deeptrace.forensics.video import analyze_video, iter_frames
from deeptrace.models.inference import DeepTraceDetector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "video"))
    p.add_argument("--config", default=None, help="Part 2 config (default configs/part2_forensics.yaml)")
    p.add_argument("--run-name", default="mixed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_forensics_config(a.config)
    out = output_dir(a.out_dir)
    path = Path(a.video)
    det = DeepTraceDetector.from_run(a.run_name, root=a.root, device=cfg["run"]["device"], checkpoint=a.checkpoint)
    t = time.time()

    def show(f):
        fp = "no face" if f["fake_probability"] is None else f"{f['prediction']} fake_p={f['fake_probability']:.4f}"
        print(f"  frame {f['frame']:5d}  t={f['time_s']:7.3f}s  {fp}")

    rep = analyze_video(det, path, cfg, on_frame=show)
    rep["video"]["file"] = path.name
    (out / f"{path.stem}_video.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for k in rep["key_frames"]:
        img = next(iter_frames(path, cfg["video"]["sample_fps"], 1, only=[k["frame"]]))[2]
        e = explain_image(det, img, **cfg["attribution"])
        title = f"frame {k['frame']} (t={k['time_s']}s): {e.result['prediction']}"
        render_overlay(e, localize(e, **cfg["localization"]), img, **cfg["overlay"],
                       title=title).save(out / f"{path.stem}_frame{k['frame']}_overlay.png")

    print(f"status {rep['status']} | {rep['usable_face_frames']}/{rep['sampling']['sampled_frames']} frames with a face "
          f"({rep['usable_face_fraction']:.0%})")
    if rep["status"] == "OK":
        ts, ess = rep["temporal_stability"], rep["evidence_stability"]
        print(f"video: {rep['prediction']} | {rep['aggregation']['summary']}")
        print("temporal stability: " + ("n/a (fewer than 2 face frames)" if ts is None else
              f"{ts['value']:.2f} ({ts['steady_pairs']}/{ts['pairs']} steady pairs, {ts['label_flips']} label flips, "
              f"std {ts['fake_probability_std']:.3f}, {len(ts['face_jumps'])} face jumps)"))
        print(f"suspicious intervals: {rep['suspicious_intervals'] or 'none'} | isolated suspicious frames: "
              f"{len(rep['isolated_suspicious_frames'])}")
        print(ess["summary"] if ess else "evidence stability: n/a")
        for k in rep["key_frames"]:
            e = k["stability"]["evidence_stability"]
            print(f"  key frame {k['frame']} t={k['time_s']}s fake_p={k['fake_probability']:.4f}: ESS {e['score']:.2f} "
                  f"{e['band']} (driver {e['driver']})")
            for o in k["robustness_observations"]:
                print(f"      - {o['message']}")
    for o in rep["observations"]:
        print(f"observation: {o['message']}")
    for w in rep["warnings"]:
        print(f"warning: {w['message']}")
    print(f"{time.time() - t:.1f}s | outputs written to {out}")


if __name__ == "__main__":
    main()
