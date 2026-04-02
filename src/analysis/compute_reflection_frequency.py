#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


CHECK_WORDS = [
    "verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect",
    "let me check", "seems right", "re-check", "double check", "double-check",
    "check again", "reconsider", "sanity check", "confirm", "validate",
]
CHECK_PREFIX = ["wait", "hmm", "let's check", "let me check"]
SWITCH_WORDS = [
    "another way", "another approach", "different approach", "another method",
    "another solution", "another strategy", "try a different", "change approach", "switch",
]
SWITCH_PREFIX = ["alternatively"]

THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)


def extract_think_text(raw: str) -> str:
    m_open = THINK_OPEN_RE.search(raw)
    if not m_open:
        return raw
    start = m_open.end()
    m_close = THINK_CLOSE_RE.search(raw, start)
    end = m_close.start() if m_close else len(raw)
    return raw[start:end]


def paragraph_steps(text: str):
    t = text.replace("\r\n", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return [s.strip() for s in t.split("\n\n") if s.strip()]


def count_rte_response_only(response_text: str):
    think = extract_think_text(response_text or "")
    steps = paragraph_steps(think.lower())
    r = t = 0
    for s in steps:
        if any(s.startswith(p) for p in CHECK_PREFIX) or any(w in s for w in CHECK_WORDS):
            r += 1
        elif any(s.startswith(p) for p in SWITCH_PREFIX) or any(w in s for w in SWITCH_WORDS):
            t += 1
    total = len(steps)
    e = max(0, total - r - t)
    return r, t, e, total


def normalize_response(model_generation):
    if isinstance(model_generation, list):
        if not model_generation:
            return ""
        first = model_generation[0]
        return first if isinstance(first, str) else str(first)
    if isinstance(model_generation, str):
        return model_generation
    return ""


def summarize_predictions(pred_path: Path):
    n = r = t = e = s = 0
    with pred_path.open() as fin:
        for line in fin:
            row = json.loads(line)
            rr, tt, ee, ss = count_rte_response_only(normalize_response(row.get("model_generation")))
            n += 1
            r += rr
            t += tt
            e += ee
            s += ss
    return {
        "N": n,
        "R": r,
        "T": t,
        "E": e,
        "S": s,
        "R_per_eg": (r / n) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("CONDITION", "PREDICTIONS_JSONL"),
        required=True,
        help="Condition label and predictions.jsonl path",
    )
    args = ap.parse_args()

    rows = []
    for condition, pred in args.run:
        row = {"condition": condition}
        row.update(summarize_predictions(Path(pred)))
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["condition", "N", "R", "T", "E", "S", "R_per_eg"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
