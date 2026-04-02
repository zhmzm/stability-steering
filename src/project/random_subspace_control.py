#!/usr/bin/env python3
"""Build random-subspace projected steering vectors as controls.

For each of N_SEEDS random seeds, generate a random k-dimensional subspace
(via QR of a Gaussian matrix), project the raw steering vectors onto its
nullspace, and save the resulting v_rand.

Usage:
    python random_subspace_control.py \
        --steering_vectors .../d_raw.l20.pt \
        --k 4 --n_seeds 10 \
        --out_dir .../random_control
"""
import argparse, os, json
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steering_vectors", required=True, help="Raw per-example (M,D) steering vectors")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    D_raw = torch.load(args.steering_vectors, weights_only=False)  # (M, D)
    M, D = D_raw.shape
    print(f"[INFO] Steering vectors: {D_raw.shape}")

    results = []
    for seed in range(args.n_seeds):
        rng = torch.Generator().manual_seed(seed)
        W = torch.randn(D, args.k, generator=rng)
        Q, _ = torch.linalg.qr(W)  # (D, k) orthonormal
        V_rand = Q  # (D, k)

        # Project: d_i_perp = d_i - V_rand (V_rand^T d_i)
        projections = D_raw @ V_rand  # (M, k)
        D_content = projections @ V_rand.T  # (M, D)
        D_perp = D_raw - D_content

        v_rand = D_perp.mean(dim=0)
        v_rand_unit = v_rand / (v_rand.norm() + 1e-12)

        out_path = os.path.join(args.out_dir, f"v_rand_seed{seed}.l{args.layer}.unit.pt")
        torch.save(v_rand_unit, out_path)

        frac = D_content.norm(dim=1).mean().item() / (D_raw.norm(dim=1).mean().item() + 1e-12)
        results.append({"seed": seed, "frac_removed_mean": frac, "path": out_path})
        print(f"  seed {seed}: frac_removed={frac:.4f}")

    json.dump(results, open(os.path.join(args.out_dir, "random_control_meta.json"), "w"), indent=2)
    print(f"[DONE] Saved {args.n_seeds} random-subspace vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
