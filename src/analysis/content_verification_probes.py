#!/usr/bin/env python3
"""Content Verification Test: linear probes on decomposed hidden state components.

Decomposes each hidden state h into:
  h^∥ = V_k V_k^T h   (content-subspace component)
  h^⊥ = (I - V_k V_k^T) h   (nullspace component)

Then trains logistic regression probes to predict:
  - Content label (MATH subject, 5 classes) from h^∥ vs h^⊥ vs h
  - Behavior label (reflection vs non-reflection) from h^∥ vs h^⊥ vs h

Usage:
    python content_verification_probes.py \
        --hidden_path .../baseline/hidden_mv/hidden.pt \
        --V_k_path .../content_subspace/V_k4.l20.pt \
        --data_path .../splits/math_train100_seed42.jsonl \
        --layer 20 \
        --out_path .../content_subspace/probe_results.json
"""
import argparse, json, os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def load_boundary_states(hidden_path, layer, data_path):
    """Load h+ (reflection) and h- (non-reflection) states from hidden.pt.

    Returns: list of (h_plus, h_minus, subject) tuples.
    """
    H = torch.load(hidden_path, weights_only=False)
    layer_dict = H[layer]

    # Load subjects from data
    subjects = []
    for line in open(data_path):
        d = json.loads(line)
        subjects.append(d.get("type", d.get("subject", "unknown")))

    states = []
    for k in sorted(layer_dict.keys()):
        step = layer_dict[k]["step"]  # (S, D)
        check_idx = set(layer_dict[k]["check_index"].tolist())  # reflection indices

        if k >= len(subjects):
            break
        subj = subjects[k]

        S = step.shape[0]
        for idx in range(S):
            is_refl = idx in check_idx
            states.append({
                "h": step[idx].float(),
                "is_reflection": is_refl,
                "subject": subj,
                "example_id": k,
            })

    return states


def run_probe(X, y, n_splits=5):
    """Run stratified k-fold logistic regression. Returns mean accuracy."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(set(y_enc)) < 2:
        return 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y_enc):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  multi_class="multinomial")
        clf.fit(X[train_idx], y_enc[train_idx])
        acc = clf.score(X[test_idx], y_enc[test_idx])
        accs.append(acc)

    return float(np.mean(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_path", required=True, help="hidden.pt from hidden_analysis.py")
    p.add_argument("--V_k_path", required=True, help="V_k.l20.pt from build_content_subspace.py")
    p.add_argument("--data_path", required=True, help="Split JSONL for subject labels")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    # Load content subspace
    V_k = torch.load(args.V_k_path, weights_only=False)  # (D, k)
    k = V_k.shape[1]
    print(f"[INFO] Content subspace rank: {k}")

    # Load boundary states
    states = load_boundary_states(args.hidden_path, args.layer, args.data_path)
    print(f"[INFO] Loaded {len(states)} hidden states")

    # Decompose each state
    h_all = torch.stack([s["h"] for s in states])  # (N_states, D)
    h_parallel = h_all @ V_k @ V_k.T  # (N_states, D)  — content component
    h_perp = h_all - h_parallel  # (N_states, D)  — nullspace component

    # Prepare labels
    behavior_labels = ["reflection" if s["is_reflection"] else "non-reflection" for s in states]
    subject_labels = [s["subject"] for s in states]

    # Convert to numpy
    h_all_np = h_all.numpy()
    h_par_np = h_parallel.numpy()
    h_perp_np = h_perp.numpy()

    results = {}

    # --- Content probe (MATH subject, 5 classes) ---
    print("\n--- Content Probe (MATH subject) ---")
    for name, X in [("h_full", h_all_np), ("h_parallel", h_par_np), ("h_perp", h_perp_np)]:
        acc = run_probe(X, subject_labels)
        results[f"content_probe_{name}"] = acc
        print(f"  {name}: {acc:.4f}")

    # --- Behavior probe (reflection vs non-reflection) ---
    print("\n--- Behavior Probe (reflection vs non-reflection) ---")
    for name, X in [("h_full", h_all_np), ("h_parallel", h_par_np), ("h_perp", h_perp_np)]:
        acc = run_probe(X, behavior_labels)
        results[f"behavior_probe_{name}"] = acc
        print(f"  {name}: {acc:.4f}")

    # --- Summary ---
    content_sep = results["content_probe_h_parallel"] - results["content_probe_h_perp"]
    behavior_sep = results["behavior_probe_h_perp"] - results["behavior_probe_h_parallel"]

    results["content_separation_pp"] = content_sep * 100
    results["behavior_separation_pp"] = behavior_sep * 100
    results["content_verification_pass"] = (content_sep >= 0.10 and behavior_sep >= 0.10)

    print(f"\n--- Verification ---")
    print(f"  Content separation: {content_sep*100:.1f}pp (need ≥10pp)")
    print(f"  Behavior separation: {behavior_sep*100:.1f}pp (need ≥10pp)")
    print(f"  PASS: {results['content_verification_pass']}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    json.dump(results, open(args.out_path, "w"), indent=2)
    print(f"\n[DONE] Saved probe results to {args.out_path}")


if __name__ == "__main__":
    main()
