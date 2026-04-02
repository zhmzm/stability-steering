#!/usr/bin/env python3
"""Compute fraction removed ||d_i - d_i^perp|| / ||d_i|| per behavior type.

Usage:
    python fraction_removed_analysis.py \
        --content_subspace_dir .../content_subspace \
        --k 4 --layer 20 \
        --out_path .../fraction_removed.json
"""
import argparse, json, os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--content_subspace_dir", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    frac_path = os.path.join(args.content_subspace_dir, f"frac_removed_k{args.k}.l{args.layer}.pt")
    if os.path.exists(frac_path):
        frac = torch.load(frac_path, weights_only=False)
        result = {
            "behavior": "reflection",
            "k": args.k,
            "layer": args.layer,
            "n_examples": len(frac),
            "frac_removed_mean": frac.mean().item(),
            "frac_removed_std": frac.std().item(),
            "frac_removed_min": frac.min().item(),
            "frac_removed_max": frac.max().item(),
            "frac_removed_median": frac.median().item(),
        }
        print(f"Reflection: mean={result['frac_removed_mean']:.4f} ± {result['frac_removed_std']:.4f}")
    else:
        # Compute from raw + V_k
        V_k = torch.load(os.path.join(args.content_subspace_dir, f"V_k{args.k}.l{args.layer}.pt"), weights_only=False)
        d_raw = torch.load(os.path.join(args.content_subspace_dir, f"d_raw.l{args.layer}.pt"), weights_only=False)

        d_content = (d_raw @ V_k) @ V_k.T
        frac = d_content.norm(dim=1) / (d_raw.norm(dim=1) + 1e-12)

        result = {
            "behavior": "reflection",
            "k": args.k,
            "layer": args.layer,
            "n_examples": len(frac),
            "frac_removed_mean": frac.mean().item(),
            "frac_removed_std": frac.std().item(),
        }
        print(f"Reflection: mean={result['frac_removed_mean']:.4f}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    json.dump(result, open(args.out_path, "w"), indent=2)
    print(f"[DONE] Saved to {args.out_path}")


if __name__ == "__main__":
    main()
