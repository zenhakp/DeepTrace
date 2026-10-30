"""Ask one question about one evidence JSON file; the answer is grounded on that file's evidence only.

  ollama serve                       (in a separate terminal first)
  python scripts/ask_evidence.py docs/part2_evidence_examples/example_image.json "Why is the stability MEDIUM?"
Prints the answer, where it came from (LLM or a fixed text), the sections and facts it used, and any rejected
problems. Writes nothing. Quote the question and paths with spaces.
"""
import argparse
import sys

from deeptrace.app.evidence import EvidenceError, load_evidence
from deeptrace.app.qa import QuestionError, ask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="evidence JSON file")
    p.add_argument("question", nargs="+", help="the question (quote it)")
    a = p.parse_args()
    try:
        answer = ask(load_evidence(a.file), " ".join(a.question))
    except (EvidenceError, QuestionError) as e:
        sys.exit(f"FAILED [{e.code}] {e}")
    used = f"LLM {answer.model}, {answer.seconds} s" if answer.source == "llm" else "fixed text"
    print(f"Q: {answer.question}\nA: {answer.text}\n[{used}; sections: {', '.join(answer.sections) or 'none'}]")
    for n in answer.notes:
        print(f"NOTE: {n}")
    if answer.facts:
        print("Evidence used:")
        for f in answer.facts:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
