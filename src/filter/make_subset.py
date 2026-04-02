#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--sample_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.source)
    rows = [json.loads(line) for line in src.open()]
    if args.sample_size > len(rows):
        raise ValueError(f"sample_size={args.sample_size} exceeds dataset size={len(rows)}")

    rng = random.Random(args.seed)
    qids = sorted(rng.sample(range(len(rows)), args.sample_size))
    subset = [rows[qid] for qid in qids]

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as fout:
        for row in subset:
            fout.write(json.dumps(row) + "\n")

    meta = {
        "source": str(src),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "qids": qids,
        "note": "Sampled from MATH train only; no examples come from MATH500.",
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[OK] wrote subset jsonl -> {out_jsonl}")
    print(f"[OK] wrote subset meta  -> {out_meta}")


if __name__ == "__main__":
    main()
