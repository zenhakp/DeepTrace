"""Stability components and Evidence Stability Score from saved robustness reports (no model needed, instant).

  python scripts/stability_report.py --robustness-dir "$HOME/Downloads/deeptrace_outputs/robustness"
Reads every *_robustness.json written by scripts/robustness_image.py and writes <stem>_stability.json,
stability_summary.md (table + weight sensitivity) and stability_sensitivity.json into --out-dir
(default ~/Downloads/deeptrace_outputs/stability; outside the repository).
The ESS is project-defined and not a probability of manipulation.
"""
import argparse
import json
from pathlib import Path

from deeptrace.forensics import DEFAULT_OUT_DIR, load_forensics_config, output_dir
from deeptrace.forensics.stability import NOTE, stability, weight_sensitivity


def _num_key(path: Path):
    head = path.stem.split("_")[0]
    return (0, int(head), path.stem) if head.isdigit() else (1, 0, path.stem)


def _row(image, rep, st) -> str:
    b = rep["baseline"]
    if st["status"] != "OK":
        return f"| {image} | {b['status']} |" + " n/a |" * 8 + " NOT_APPLICABLE | n/a | n/a | n/a |"
    e, ps = st["evidence_stability"], st["prediction_stability"]
    capped = f"yes (from {e['band_uncapped']})" if e["band_capped"] else "no"
    driver = f"{e['driver']} ({e['points_lost'][e['driver']]:.2f})" if e["driver"] else "none"
    diag = "; ".join(f"{m} {d['evidence_stability']['score']:.2f} {d['evidence_stability']['band']}"
                     for m, d in st["diagnostics"].items()) or "n/a"
    return (f"| {image} | {b['prediction']} | {b['fake_probability']:.4f} | {ps['flips'] + ps['no_face']} / {ps['runs']} "
            f"| {ps['value']:.2f} | {st['confidence_stability']['value']:.2f} "
            f"| {st['localization_stability']['value']:.2f} | **{e['score']:.2f}** | {e['band']} | {capped} "
            f"| {driver} | {diag} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robustness-dir", default=str(DEFAULT_OUT_DIR / "robustness"))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "stability"))
    p.add_argument("--config", default=None, help="Part 2 config (default configs/part2_forensics.yaml)")
    a = p.parse_args()

    cfg = load_forensics_config(a.config)["stability"]
    src = Path(a.robustness_dir).expanduser()
    files = sorted(src.glob("*_robustness.json"), key=_num_key)
    if not files:
        raise FileNotFoundError(f"No *_robustness.json files in {src}. Run scripts/robustness_image.py first.")
    out = output_dir(a.out_dir)
    lines = [f"# Evidence Stability Score (measured on {len(files)} images: a small sample, not representative)", "",
             NOTE, "", f"Settings: `{json.dumps({k: v for k, v in cfg.items() if k != 'sensitivity_weights'})}`", "",
             "| image | prediction | fake_p | flips / runs | prediction | confidence | localization | ESS | band "
             "| band capped | driver (cost) | fixed_crop ESS (diagnostic) |", "|" + "---|" * 12]
    results = []
    for f in files:
        rep = json.loads(f.read_text(encoding="utf-8"))
        st = stability(rep, cfg["mode"], cfg["weights"], cfg["bands"], cfg["margin_floor"], cfg["cap_band"])
        image = rep.get("image", f.stem.replace("_robustness", ""))
        results.append({"image": image, **st})
        (out / f"{Path(image).stem}_stability.json").write_text(json.dumps({"image": image, **st}, indent=2),
                                                                 encoding="utf-8")
        lines.append(_row(image, rep, st))
        print(f"{image}: " + (st["evidence_stability"]["summary"] if st["status"] == "OK"
                              else f"NOT_APPLICABLE ({st['reason']})"))

    sens = weight_sensitivity(results, cfg["sensitivity_weights"], cfg["bands"], cfg["cap_band"])
    lines += ["", "## Weight sensitivity (same components, alternative weights, band cap applied)", "",
              f"Chosen weights: `{json.dumps(cfg['weights'])}`.", "",
              "| alternative | prediction / confidence / localization | photos changing band | changes |",
              "|---|---|---|---|"]
    for row in sens:
        w = row["weights"]
        changes = ", ".join(f"{c['image']} {c['band']} -> {c['alt_band']} ({c['score']:.2f} -> {c['alt_score']:.2f})"
                            for c in row["changes"]) or "none"
        lines.append(f"| {row['name']} | {w['prediction']} / {w['confidence']} / {w['localization']} "
                     f"| {row['changed']} of {row['images']} | {changes} |")
        print(f"sensitivity {row['name']}: {row['changed']} of {row['images']} photos change band")
    (out / "stability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "stability_sensitivity.json").write_text(json.dumps(sens, indent=2), encoding="utf-8")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
