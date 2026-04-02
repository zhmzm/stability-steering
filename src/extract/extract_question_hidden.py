#!/usr/bin/env python3
"""Extract mean-pooled question-token hidden states at a given layer.

For each example in the split, encodes the question text alone and extracts
hidden states for ALL question tokens (not just the last one), then mean-pools
to produce q_i ∈ R^D.

Usage:
    python extract_question_hidden.py \
        --data_dir  .../baseline \
        --data_path .../splits/math_train100_seed42.jsonl \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --layer 20 \
        --out_path  .../question_hidden_l20.pt
"""
import argparse, json, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_questions(data_path: str):
    """Load raw question texts from the split JSONL."""
    questions = []
    subjects = []
    for line in open(data_path):
        d = json.loads(line)
        questions.append(d["problem"].strip())
        subjects.append(d.get("type", d.get("subject", "unknown")))
    return questions, subjects


@torch.no_grad()
def extract_question_states(model, tok, question: str, layer: int) -> torch.Tensor:
    """Get hidden states for all tokens of the question at the given layer.

    Returns: (T, D) tensor of hidden states.
    """
    text = f"Question: {question}"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[layer][0]  # (T, D)
    return h.detach().cpu().to(torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="Split JSONL (e.g., math_train100_seed42.jsonl)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--out_path", required=True, help="Output .pt file for question hidden states")
    args = p.parse_args()

    questions, subjects = load_questions(args.data_path)
    N = len(questions)
    print(f"[INFO] Loaded {N} questions from {args.data_path}")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    q_means = []  # per-example mean-pooled question hidden states
    q_subjects = []

    for i, (question, subject) in enumerate(zip(questions, subjects)):
        h = extract_question_states(model, tok, question, args.layer)  # (T, D)
        q_i = h.mean(dim=0)  # (D,)
        q_means.append(q_i)
        q_subjects.append(subject)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{N}] done")

    Q = torch.stack(q_means, dim=0)  # (N, D)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save({
        "Q": Q,
        "subjects": q_subjects,
        "layer": args.layer,
        "n_examples": N,
    }, args.out_path)
    print(f"[DONE] Saved Q shape {Q.shape} to {args.out_path}")


if __name__ == "__main__":
    main()
