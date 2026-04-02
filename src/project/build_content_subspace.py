#!/usr/bin/env python3
"""Build content subspace via SVD on centered question-token hidden states.

Given the Q matrix (N, D) of mean-pooled question-token hidden states:
  1. Center: Q_c = Q - mean(Q)
  2. SVD: Q_c = U Σ V^T
  3. Content subspace: top-k right singular vectors
  4. Save V_k, singular values, and the nullspace projector

Also projects per-example steering vectors d_i and saves v_proj.

Usage:
    python build_content_subspace.py \
        --question_hidden .../question_hidden_l20.pt \
        --steering_vectors .../vector_per_example_mv/layer_20_transition_reflection_steervec.pt \
        --k 4 \
        --out_dir .../content_subspace
"""
import argparse, os, json
import torch
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question_hidden", required=True, help="question_hidden_l20.pt from extract_question_hidden.py")
    p.add_argument("--steering_vectors", required=True, help="Per-example (N,D) steering vectors .pt")
    p.add_argument("--steering_manifest", default=None, help="Per-example manifest .json (optional)")
    p.add_argument("--k", type=int, default=4, help="Rank of content subspace")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load question hidden states ---
    qh = torch.load(args.question_hidden, weights_only=False)
    Q_raw = qh["Q"]  # (N, D)
    subjects = qh.get("subjects", [])
    N, D = Q_raw.shape
    print(f"[INFO] Q_raw shape: {Q_raw.shape}")

    # --- Center ---
    Q_mean = Q_raw.mean(dim=0, keepdim=True)  # (1, D)
    Q_c = Q_raw - Q_mean  # (N, D)

    # --- SVD ---
    U, S, Vt = torch.linalg.svd(Q_c, full_matrices=False)  # U: (N,N), S: (N,), Vt: (N,D)
    print(f"[INFO] SVD done. Top-10 singular values: {S[:10].tolist()}")

    # Content subspace: top-k right singular vectors
    V_k = Vt[:args.k].T  # (D, k) — columns are the content directions
    print(f"[INFO] Content subspace V_k shape: {V_k.shape} (rank {args.k})")

    # Nullspace projector: P_perp = I - V_k V_k^T
    # We don't store the full (D,D) matrix — apply it as needed.

    # --- Load per-example steering vectors ---
    D_raw = torch.load(args.steering_vectors, weights_only=False)  # (M, D)
    M = D_raw.shape[0]
    print(f"[INFO] Steering vectors shape: {D_raw.shape}")

    # --- Project each d_i onto nullspace ---
    # d_i_perp = d_i - V_k (V_k^T d_i)
    projections = D_raw @ V_k  # (M, k) — projections onto content subspace
    D_content = projections @ V_k.T  # (M, D) — content component
    D_perp = D_raw - D_content  # (M, D) — nullspace component

    # Compute fraction removed per example
    frac_removed = D_content.norm(dim=1) / (D_raw.norm(dim=1) + 1e-12)
    print(f"[INFO] Fraction removed: mean={frac_removed.mean():.4f}, "
          f"std={frac_removed.std():.4f}, min={frac_removed.min():.4f}, max={frac_removed.max():.4f}")

    # --- Aggregate projected vectors ---
    v_proj = D_perp.mean(dim=0)  # (D,)
    v_proj_unit = v_proj / (v_proj.norm() + 1e-12)

    # Also compute SEAL vector (unprojected mean) for comparison
    v_seal = D_raw.mean(dim=0)
    v_seal_unit = v_seal / (v_seal.norm() + 1e-12)

    # Cosine similarity between v_proj and v_seal
    cos_sim = torch.dot(v_proj_unit, v_seal_unit).item()
    print(f"[INFO] Cosine(v_proj, v_seal) = {cos_sim:.4f}")

    # --- Save everything ---
    torch.save(V_k, os.path.join(args.out_dir, f"V_k{args.k}.l{args.layer}.pt"))
    torch.save(S, os.path.join(args.out_dir, f"singular_values.l{args.layer}.pt"))
    torch.save(v_proj_unit, os.path.join(args.out_dir, f"v_proj_k{args.k}.l{args.layer}.unit.pt"))
    torch.save(v_proj, os.path.join(args.out_dir, f"v_proj_k{args.k}.l{args.layer}.pt"))
    torch.save(v_seal_unit, os.path.join(args.out_dir, f"v_seal.l{args.layer}.unit.pt"))
    torch.save(v_seal, os.path.join(args.out_dir, f"v_seal.l{args.layer}.pt"))
    torch.save(D_perp, os.path.join(args.out_dir, f"d_perp_k{args.k}.l{args.layer}.pt"))
    torch.save(D_content, os.path.join(args.out_dir, f"d_content_k{args.k}.l{args.layer}.pt"))
    torch.save(D_raw, os.path.join(args.out_dir, f"d_raw.l{args.layer}.pt"))
    torch.save(frac_removed, os.path.join(args.out_dir, f"frac_removed_k{args.k}.l{args.layer}.pt"))

    # Save metadata
    meta = {
        "k": args.k,
        "layer": args.layer,
        "n_questions": N,
        "n_steering_vectors": M,
        "D": D,
        "top_10_singular_values": S[:10].tolist(),
        "frac_removed_mean": frac_removed.mean().item(),
        "frac_removed_std": frac_removed.std().item(),
        "cos_v_proj_v_seal": cos_sim,
        "subjects": subjects,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)

    print(f"[DONE] Content subspace + projected vectors saved to {args.out_dir}")
    print(f"  v_proj: {os.path.join(args.out_dir, f'v_proj_k{args.k}.l{args.layer}.unit.pt')}")
    print(f"  v_seal: {os.path.join(args.out_dir, f'v_seal.l{args.layer}.unit.pt')}")


if __name__ == "__main__":
    main()
