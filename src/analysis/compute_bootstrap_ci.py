#!/usr/bin/env python3
"""Compute paired bootstrap 95% CI on accuracy difference between two systems.

Usage:
    python compute_bootstrap_ci.py \
        --eval_a .../v_proj/coef_-100.0/base_remove_bos/math_eval.jsonl \
        --eval_b .../seal/coef_-100.0/base_remove_bos/math_eval.jsonl \
        --n_bootstrap 10000 \
        --out_path .../bootstrap_ci.json
"""
import argparse, json, os
import numpy as np


def load_correctness(eval_path):
    """Load per-example correctness from math_eval.jsonl."""
    correct = []
    for line in open(eval_path):
        d = json.loads(line)
        correct.append(int(d.get("correct", d.get("is_correct", 0))))
    return np.array(correct)


def paired_bootstrap_ci(correct_a, correct_b, n_bootstrap=10000, alpha=0.05, seed=42):
    """Compute paired bootstrap CI on accuracy difference (A - B)."""
    rng = np.random.RandomState(seed)
    n = len(correct_a)
    assert len(correct_b) == n

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        acc_a = correct_a[idx].mean()
        acc_b = correct_b[idx].mean()
        diffs.append(acc_a - acc_b)

    diffs = np.array(diffs)
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    mean_diff = diffs.mean()

    return {
        "mean_diff": float(mean_diff),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_examples": int(n),
        "acc_a": float(correct_a.mean()),
        "acc_b": float(correct_b.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_a", required=True, help="math_eval.jsonl for system A (v_proj)")
    p.add_argument("--eval_b", required=True, help="math_eval.jsonl for system B (SEAL)")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    ca = load_correctness(args.eval_a)
    cb = load_correctness(args.eval_b)

    result = paired_bootstrap_ci(ca, cb, n_bootstrap=args.n_bootstrap)

    print(f"System A accuracy: {result['acc_a']:.4f}")
    print(f"System B accuracy: {result['acc_b']:.4f}")
    print(f"Difference (A-B): {result['mean_diff']:.4f}")
    print(f"95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"Excludes zero: {result['excludes_zero']}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    json.dump(result, open(args.out_path, "w"), indent=2)
    print(f"[DONE] Saved to {args.out_path}")


if __name__ == "__main__":
    main()
