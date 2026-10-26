"""Full Part 2 analysis: one structured evidence JSON per image or video (the Part 3 contract).

  python scripts/analyze.py --input "$HOME/Downloads/samples/1.jpg"
  python scripts/analyze.py --input "$HOME/Downloads/samples/clip.mp4" --out-dir "$HOME/Downloads/deeptrace_outputs/evidence"
Writes <stem>_evidence.json plus overlay PNG(s) into --out-dir (default ~/Downloads/deeptrace_outputs/evidence;
must be outside the repository). Every document is checked against docs/part2_evidence_schema.json before it is
written. Quote paths with spaces.
"""
import argparse
import json
import time
from pathlib import Path

from deeptrace.forensics import DEFAULT_OUT_DIR, load_forensics_config, output_dir
from deeptrace.forensics.attribution import explain_image
from deeptrace.forensics.evidence import SCHEMA_PATH, image_evidence, media_type, validate, video_evidence
from deeptrace.forensics.localization import localize
from deeptrace.forensics.overlay import render_overlay
from deeptrace.forensics.video import iter_frames
from deeptrace.models.inference import DeepTraceDetector


def evidence_for(det, path: Path, cfg, config_path, out: Path):
    if media_type(path) == "image":
        ev, e, loc = image_evidence(det, path, cfg, config_path)
        name = f"{path.stem}_overlay.png"
        render_overlay(e, loc, DeepTraceDetector._load(path), **cfg["overlay"]).save(out / name)
        ev["artifacts"] = {"overlay": name}
        return ev
    ev = video_evidence(det, path, cfg, config_path)
    overlays = {}
    for k in ev["robustness"]["key_frames"] if ev["robustness"] else []:
        img = next(iter_frames(path, cfg["video"]["sample_fps"], 1, only=[k]))[2]
        e = explain_image(det, img, **cfg["attribution"])
        name = f"{path.stem}_frame{k}_overlay.png"
        render_overlay(e, localize(e, **cfg["localization"]), img, **cfg["overlay"],
                       title=f"frame {k}: {e.result['prediction']}").save(out / name)
        overlays[str(k)] = name
    ev["artifacts"] = {"key_frame_overlays": overlays}
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, nargs="+", help="image or video file(s)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "evidence"))
    p.add_argument("--config", default=None, help="Part 2 config (default configs/part2_forensics.yaml)")
    p.add_argument("--run-name", default="mixed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_forensics_config(a.config)
    out = output_dir(a.out_dir)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    det = DeepTraceDetector.from_run(a.run_name, root=a.root, device=cfg["run"]["device"], checkpoint=a.checkpoint)
    for item in a.input:
        path = Path(item)
        t = time.time()
        ev = evidence_for(det, path, cfg, a.config, out)
        errors = validate(ev, schema)
        if errors:
            raise RuntimeError(f"Evidence for {path.name} does not match the schema: " + "; ".join(errors[:10]))
        target = out / f"{path.stem}_evidence.json"
        target.write_text(json.dumps(ev, indent=2), encoding="utf-8")
        ess = ev["evidence_stability"]
        fp = "n/a" if ev["fake_probability"] is None else f"{ev['fake_probability']:.4f}"
        es = "n/a" if ess is None else f"{ess['score']:.2f} {ess['band']}"
        print(f"{path.name}: {ev['media_type']} {ev['status']} {ev['prediction']} fake_p={fp} | "
              f"{len(ev['suspicious_regions'])} regions | ESS {es} | schema OK | {time.time() - t:.1f}s -> {target.name}")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
