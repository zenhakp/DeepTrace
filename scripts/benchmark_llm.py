"""Measure local LLM candidates on the example evidence: speed, RAM and crude wording red flags.

  ollama serve                       (in a separate terminal first)
  python scripts/benchmark_llm.py --models qwen3:4b-instruct llama3.2:3b
Writes every answer plus results.json to --out-dir (default ~/Downloads/deeptrace_outputs/llm_benchmark; must be
outside the repository) so the texts can be read, not just counted. Each model is unloaded after its runs.

The prompt here is an INTERIM one (a short fact list + strict rules) to measure models on a realistic prompt size;
step 3 replaces it with the real evidence digest and grounded prompt, and step 4 adds the real rule checks. The red
flags below are crude string checks, not the step 4 validator.
"""
import argparse
import json
import re
from pathlib import Path

from deeptrace.app.evidence import REPO_ROOT, load_evidence
from deeptrace.app.llm import LLMSettings, LLMUnavailable, chat, check, loaded, unload

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"
SYSTEM = ("You explain the output of a deepfake detector to a non-expert. Use ONLY the facts given. Do not add any "
          "verdict, number, region or claim that is not in the facts. Follow the wording rule exactly. Never write "
          "'definitely', 'proves', 'certain' or 'X% chance'. Write 120-180 words of plain prose.")
BANNED = re.compile(r"\bdefinitely\b|\bcertainly\b|\bwith certainty\b|\bproves?\b|\bproof\b|\bguarantee|"
                    r"manipulated pixels|ground[- ]truth", re.I)
# Required disclaimers ("not ground-truth manipulated pixels") are removed before BANNED is applied.
DISCLAIMER = re.compile(r"\bnot\s+(?:\w+\s+){0,2}?(?:ground[- ]truth(?:\s+manipulated\s+pixels)?|manipulated\s+pixels)",
                        re.I)
PERCENT_SCORE = re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:chance|confident|confidence|likely|likelihood|probability|sure)",
                           re.I)
VERDICT = re.compile(r"\b(?:predict(?:ed|s)?|classified|flagged|detected|labell?ed)\b[^.]{0,40}?\b(?:manipulated|fake|"
                     r"real|authentic|genuine)\b|\b(?:is|was|appears to be|looks)\s+(?:manipulated|fake|real|authentic|"
                     r"genuine)\b", re.I)
HEDGE = re.compile(r"probability that|likelihood that|whether|\bnot\b|\bno\b", re.I)  # before a VERDICT match
NUMBER = re.compile(r"\d+\.\d+|\d+(?=\s*%)")


def facts(d):
    """Interim fact list (see module docstring)."""
    lines = [f"media type: {d['media_type']}"]
    if d["status"] == "NO_FACE":
        lines += ["status: NO_FACE (no face was found, so the detector made no prediction)"]
    else:
        lines += [f"detector prediction: {d['prediction']}",
                  f"fake-class probability: {d['fake_probability']:.2f}", d["evidence_stability"]["summary"]]
        if d["evidence_stability"]["band_capped"]:
            lines.append(d["evidence_stability"]["cap_reason"])
        loc = d["localization"] or {}
        lines.append(loc.get("message") or loc.get("note", ""))
        lines.append(f"number of model-indicated suspicious regions: {len(d['suspicious_regions'])}")
        if d["aggregation"]:
            lines.append(d["aggregation"]["summary"])
        for i in d["suspicious_intervals"]:
            lines.append(f"suspicious interval {i['start_s']} s to {i['end_s']} s")
        lines += [o["message"] for o in d["robustness_observations"]]
        lines += [o.get("message", "") for o in d["observations"]]
    lines += [w["message"] for w in d["warnings"]]
    lines += [f"limitation: {x}" for x in d["limitations"]]
    return "\n".join(f"- {x}" for x in lines if x)


def _known(n: str, facts_numbers) -> bool:
    """n appears in the facts, or is one of them rounded to fewer decimals (0.6518 -> 0.65)."""
    places = len(n.split(".")[1]) if "." in n else 0
    return any(round(float(k), places) == float(n) for k in facts_numbers)


def red_flags(text, d, fact_text):
    """Crude checks (see module docstring). Numbers are compared with the facts only, NOT with score_semantics:
    its example number (0.91) is wording guidance, not evidence, and one model presented it as a NO_FACE result."""
    flags = [f"banned:{m.group(0)!r}" for m in BANNED.finditer(DISCLAIMER.sub(" ", text))]
    flags += [f"percent_score:{m.group(0)!r}" for m in PERCENT_SCORE.finditer(text)]
    if d["status"] == "NO_FACE":
        flags += [f"verdict_on_no_face:{m.group(0)!r}" for m in VERDICT.finditer(text)
                  if not HEDGE.search(text[max(0, m.start() - 50):m.start()])]
    facts_numbers = NUMBER.findall(fact_text)
    flags += [f"number_not_in_facts:{n}" for n in NUMBER.findall(text) if not _known(n, facts_numbers)]
    return flags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["qwen3:4b-instruct", "llama3.2:3b"])
    p.add_argument("--evidence", nargs="+", default=[str(EXAMPLES / f"example_{n}.json")
                                                      for n in ("image", "video", "no_face")])
    p.add_argument("--out-dir", default=str(Path.home() / "Downloads" / "deeptrace_outputs" / "llm_benchmark"))
    a = p.parse_args()
    out = Path(a.out_dir).expanduser().resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise SystemExit(f"--out-dir is inside the repository ({out}); choose a folder outside it")
    out.mkdir(parents=True, exist_ok=True)
    docs = [load_evidence(f) for f in a.evidence]

    results = []
    print(f"{'model':<20} {'evidence':<24} {'in tok':>6} {'out tok':>7} {'load s':>6} {'in tok/s':>8} "
          f"{'out tok/s':>9} {'total s':>7}  red flags")
    for model in a.models:
        s = LLMSettings.from_config(model=model)
        try:
            check(s)
        except LLMUnavailable as e:
            print(f"{model:<20} SKIPPED: {e}")
            continue
        ram = None
        for ev in docs:
            fact_text = facts(ev.data)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Wording rule: {ev.data['score_semantics']}\n\nFacts:\n{fact_text}"}]
            try:
                r = chat(msgs, s)
            except LLMUnavailable as e:
                print(f"{model:<20} {ev.source.name:<24} FAILED: {e}")
                continue
            ram = ram or next((m["size_bytes"] for m in loaded(s) if m["name"] == model), None)
            flags = red_flags(r.text, ev.data, fact_text)
            rate = lambda n, t: n / t if t else 0.0
            print(f"{model:<20} {ev.source.name:<24} {r.prompt_tokens:>6} {r.output_tokens:>7} {r.load_s:>6.1f} "
                  f"{rate(r.prompt_tokens, r.prompt_s):>8.1f} {rate(r.output_tokens, r.output_s):>9.1f} "
                  f"{r.total_s:>7.1f}  {', '.join(flags) or '-'}")
            (out / f"{model.replace(':', '_')}__{ev.source.stem}.md").write_text(r.text + "\n", encoding="utf-8")
            results.append({"model": model, "evidence": ev.source.name, "prompt_tokens": r.prompt_tokens,
                            "output_tokens": r.output_tokens, "load_s": round(r.load_s, 2),
                            "prompt_s": round(r.prompt_s, 2), "output_s": round(r.output_s, 2),
                            "total_s": round(r.total_s, 2), "words": len(r.text.split()), "red_flags": flags})
        if ram:
            print(f"{model:<20} RAM while loaded (ollama ps): {ram / 1e9:.1f} GB")
            results.append({"model": model, "ram_bytes": ram})
        unload(s)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Answers and results.json written to {out}")


if __name__ == "__main__":
    main()
