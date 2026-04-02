#!/usr/bin/env python3
"""Reasoning compression analysis: count reflections, tokens, and build Pareto data.

Usage:
    python compression_analysis.py \
        --predictions_dir .../MATH500/DeepSeek-R1-Distill-Qwen-1.5B/math_train100_seed42 \
        --conditions baseline,seal,v_proj \
        --out_path .../compression_analysis.json
"""
import argparse, json, os, re
from collections import defaultdict

REFLECTION_KEYWORDS = {"wait", "verify", "check", "alternatively", "hold on", "let me reconsider"}


def count_reflections(text):
    """Count reflection paragraphs in a reasoning trace."""
    paragraphs = text.split("\n\n")
    count = 0
    for para in paragraphs:
        first_words = para.strip().lower()[:50]
        for kw in REFLECTION_KEYWORDS:
            if kw in first_words:
                count += 1
                break
    return count


def count_think_tokens(text, tokenizer=None):
    """Count tokens within <think>...</think> blocks."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)
    else:
        think_text = text
    # Approximate token count by whitespace splitting
    return len(think_text.split())


def analyze_predictions(pred_path):
    """Analyze a predictions.jsonl file."""
    reflections = []
    token_counts = []
    correct = []

    for line in open(pred_path):
        d = json.loads(line)
        output = d.get("output", d.get("prediction", ""))
        reflections.append(count_reflections(output))
        token_counts.append(count_think_tokens(output))
        # correctness from companion math_eval.jsonl if available
        correct.append(d.get("correct", None))

    return {
        "n_examples": len(reflections),
        "mean_reflections": sum(reflections) / max(len(reflections), 1),
        "mean_tokens": sum(token_counts) / max(len(token_counts), 1),
        "reflections": reflections,
        "token_counts": token_counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_dir", required=True, help="Root dir containing condition subdirs")
    p.add_argument("--conditions", required=True, help="Comma-separated condition names")
    p.add_argument("--coefs", default=None, help="Comma-separated coefs (e.g., -100,-50). If None, uses best from metrics.json")
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    conditions = args.conditions.split(",")
    results = {}

    for cond in conditions:
        cond_dir = os.path.join(args.predictions_dir, cond)
        if not os.path.isdir(cond_dir):
            print(f"[WARN] {cond_dir} not found, skipping")
            continue

        # Find best coef or use specified
        if args.coefs:
            coefs = args.coefs.split(",")
        else:
            # Find all coef subdirs
            coefs = []
            for d in os.listdir(cond_dir):
                if d.startswith("coef_"):
                    coefs.append(d.replace("coef_", ""))
            if not coefs:
                # Try direct predictions.jsonl
                pred_path = os.path.join(cond_dir, "base_run", "base_remove_bos", "predictions.jsonl")
                if os.path.exists(pred_path):
                    results[cond] = analyze_predictions(pred_path)
                    print(f"  {cond}: R/ex={results[cond]['mean_reflections']:.3f}, "
                          f"tokens/ex={results[cond]['mean_tokens']:.1f}")
                continue

        for coef in coefs:
            pred_path = os.path.join(cond_dir, f"coef_{coef}", "base_remove_bos", "predictions.jsonl")
            if not os.path.exists(pred_path):
                pred_path = os.path.join(cond_dir, f"coef_{coef}", "predictions.jsonl")
            if os.path.exists(pred_path):
                key = f"{cond}_coef{coef}"
                results[key] = analyze_predictions(pred_path)
                # Remove per-example lists for JSON size
                results[key].pop("reflections", None)
                results[key].pop("token_counts", None)
                print(f"  {key}: R/ex={results[key]['mean_reflections']:.3f}, "
                      f"tokens/ex={results[key]['mean_tokens']:.1f}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    json.dump(results, open(args.out_path, "w"), indent=2)
    print(f"\n[DONE] Saved compression analysis to {args.out_path}")


if __name__ == "__main__":
    main()
