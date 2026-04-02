#!/usr/bin/env python3
"""Create 100-question extraction split + 50-question dev split from MATH train.

Usage:
    python create_split_100.py \
        --source claim_1/data/MATH/train.jsonl \
        --out_dir claim_ab/splits \
        --extract_n 100 --extract_seed 42 \
        --dev_n 50 --dev_seed 43
"""
import argparse, json, pathlib, random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Full MATH train.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--extract_n", type=int, default=100)
    p.add_argument("--extract_seed", type=int, default=42)
    p.add_argument("--dev_n", type=int, default=50)
    p.add_argument("--dev_seed", type=int, default=43)
    args = p.parse_args()

    lines = [json.loads(l) for l in open(args.source)]
    total = len(lines)
    all_ids = list(range(total))

    # --- extraction split (seed 42) ---
    rng_e = random.Random(args.extract_seed)
    extract_ids = sorted(rng_e.sample(all_ids, args.extract_n))

    # --- dev split (seed 43, non-overlapping) ---
    remaining = sorted(set(all_ids) - set(extract_ids))
    rng_d = random.Random(args.dev_seed)
    dev_ids = sorted(rng_d.sample(remaining, args.dev_n))

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # write extraction split
    ext_tag = f"math_train{args.extract_n}_seed{args.extract_seed}"
    with open(out / f"{ext_tag}.jsonl", "w") as f:
        for i in extract_ids:
            f.write(json.dumps(lines[i]) + "\n")
    json.dump({
        "source": str(args.source),
        "sample_size": args.extract_n,
        "seed": args.extract_seed,
        "qids": extract_ids,
        "note": "Sampled from MATH train only; no examples come from MATH500.",
    }, open(out / f"{ext_tag}.meta.json", "w"), indent=2)

    # write dev split
    dev_tag = f"math_dev{args.dev_n}_seed{args.dev_seed}"
    with open(out / f"{dev_tag}.jsonl", "w") as f:
        for i in dev_ids:
            f.write(json.dumps(lines[i]) + "\n")
    json.dump({
        "source": str(args.source),
        "sample_size": args.dev_n,
        "seed": args.dev_seed,
        "qids": dev_ids,
        "excludes": ext_tag,
        "note": "Dev split for coefficient selection. Non-overlapping with extraction split.",
    }, open(out / f"{dev_tag}.meta.json", "w"), indent=2)

    print(f"[OK] Extraction: {ext_tag} ({len(extract_ids)} examples)")
    print(f"[OK] Dev: {dev_tag} ({len(dev_ids)} examples)")
    print(f"[OK] Overlap check: {len(set(extract_ids) & set(dev_ids))} shared (should be 0)")


if __name__ == "__main__":
    main()
