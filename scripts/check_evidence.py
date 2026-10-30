"""Check evidence JSON files exactly as the Part 3 app will (schema 1.0 + Part 3 consistency rules). Reads only.

  python scripts/check_evidence.py docs/part2_evidence_examples/*.json
  python scripts/check_evidence.py "$HOME/Downloads/deeptrace_outputs/evidence/"*_evidence.json
Exit code 1 if any file fails. Quote paths with spaces.
"""
import argparse
import sys
from pathlib import Path

from deeptrace.app.evidence import EvidenceError, load_evidence


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="evidence JSON file(s)")
    files = p.parse_args().files
    failed = 0
    for f in files:
        name = Path(f).name
        try:
            ev = load_evidence(f)
        except EvidenceError as e:
            failed += 1
            print(f"{name}: FAILED [{e.code}]")
            for line in e.errors[:10]:
                print(f"    {line}")
            continue
        d = ev.data
        print(f"{name}: OK | {d['media_type']} {d['status']} prediction={d['prediction']} | "
              f"{len(d['suspicious_regions'])} region(s) | overlays {sorted(ev.overlay_files().values())}")
    print(f"{len(files) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
