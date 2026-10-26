"""Turn real evidence documents into anonymised examples for docs/part2_evidence_examples/ (for Part 3).

  python scripts/make_evidence_examples.py --image a_evidence.json --video b_evidence.json --no-face c_evidence.json
Only identifying fields change: input file name and sha256, generated_at and artifact names. All numbers are kept
exactly as the pipeline produced them, and each example is re-validated against docs/part2_evidence_schema.json.
"""
import argparse
import json
from pathlib import Path

from deeptrace.forensics import REPO_ROOT
from deeptrace.forensics.evidence import SCHEMA_PATH, validate

OUT = REPO_ROOT / "docs" / "part2_evidence_examples"
PLACEHOLDER_SHA = "0" * 64


def anonymise(ev, name: str):
    ev = json.loads(json.dumps(ev))
    ext = Path(ev["input"]["file_name"]).suffix.lower()
    ev["input"]["file_name"] = f"{name}{ext}"
    ev["input"]["sha256"] = PLACEHOLDER_SHA
    ev["generated_at"] = "2026-01-01T00:00:00+00:00"
    if "overlay" in ev["artifacts"]:
        ev["artifacts"] = {"overlay": f"{name}_overlay.png"}
    elif "key_frame_overlays" in ev["artifacts"]:
        ev["artifacts"] = {"key_frame_overlays": {k: f"{name}_frame{k}_overlay.png"
                                                  for k in ev["artifacts"]["key_frame_overlays"]}}
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--no-face", required=True)
    a = p.parse_args()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name in ((a.image, "example_image"), (a.video, "example_video"), (a.no_face, "example_no_face")):
        ev = anonymise(json.loads(Path(src).read_text(encoding="utf-8")), name)
        errors = validate(ev, schema)
        if errors:
            raise RuntimeError(f"{name} does not match the schema: " + "; ".join(errors[:10]))
        (OUT / f"{name}.json").write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
        print(f"{name}.json: {ev['media_type']} {ev['status']} {ev['prediction']} | schema OK")
    print(f"Examples written to {OUT}")


if __name__ == "__main__":
    main()
