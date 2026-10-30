"""Explain evidence JSON file(s) with the local LLM (grounded on the digest) and print every section.

  ollama serve                       (in a separate terminal first; without it the template text is used)
  python scripts/explain_evidence.py docs/part2_evidence_examples/example_image.json
  python scripts/explain_evidence.py "$HOME/Downloads/deeptrace_outputs/evidence/2_evidence.json" --no-llm
Writes <stem>_explanation.json and <stem>_report.md to --out-dir (default ~/Downloads/deeptrace_outputs/explanations;
must be outside the repository). The report links overlay images by name, so they show when it sits next to them. Each LLM text is checked against the evidence; a rejected one shows the template text and its
problems. Quote paths with spaces.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deeptrace.app.evidence import REPO_ROOT, EvidenceError, load_evidence
from deeptrace.app.explain import explain
from deeptrace.app.llm import LLMUnavailable
from deeptrace.app.report import render_markdown


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="evidence JSON file(s)")
    p.add_argument("--out-dir", default=str(Path.home() / "Downloads" / "deeptrace_outputs" / "explanations"))
    p.add_argument("--no-llm", action="store_true", help="template text only (no Ollama needed)")
    a = p.parse_args()
    out = Path(a.out_dir).expanduser().resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise SystemExit(f"--out-dir is inside the repository ({out}); choose a folder outside it")
    out.mkdir(parents=True, exist_ok=True)

    def offline(*_, **__):
        raise LLMUnavailable("not_running", "--no-llm was given")

    for f in a.files:
        try:
            ev = load_evidence(f)
        except EvidenceError as e:
            print(f"{Path(f).name}: FAILED [{e.code}] {e.errors[:3]}")
            continue
        ex = explain(ev, llm=offline) if a.no_llm else explain(ev)
        used = (f"LLM {ex.model}, {ex.seconds} s, {ex.prompt_tokens} prompt tokens processed, "
                f"{ex.output_tokens} output tokens" if ex.llm_used else "template only")
        print(f"===== {ev.source.name}: {ev.data['media_type']} {ev.data['status']} | {used}")
        for n in ex.notes:
            print(f"NOTE: {n}")
        for s in ex.sections:
            print(f"\n## {s.title} [{s.source}{', LLM text rejected' if s.problems else ''}]\n{s.text}")
        stem = ev.source.stem.replace("_evidence", "")
        target = out / f"{stem}_explanation.json"
        target.write_text(json.dumps(asdict(ex), indent=2), encoding="utf-8")
        (out / f"{stem}_report.md").write_text(render_markdown(ev, ex), encoding="utf-8")
        print(f"\n-> {target}\n-> {out / f'{stem}_report.md'}\n")


if __name__ == "__main__":
    main()
