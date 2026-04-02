#!/usr/bin/env python3
# claim_1/extract_qonly_meaning_vectors.py

import os, json, argparse, pathlib, sys
from typing import List, Dict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_eval_pairs(data_dir: str, data_path: str) -> List[Dict]:
    """Align with math_eval.jsonl ordering (same as your existing pipeline)."""
    lines = [json.loads(l) for l in open(data_path)]
    eval_lines = [json.loads(l) for l in open(os.path.join(data_dir, "math_eval.jsonl"))]
    lines = lines[:len(eval_lines)]
    out = []
    for d, e in zip(lines, eval_lines):
        out.append({
            "prompt": e["prompt"],        # full prompt used during eval generation
            "problem": d["problem"],      # raw question text
            "answer": e["answer"],        # ground truth
        })
    return out

@torch.no_grad()
def hidden_last_token(model, tok, text: str, layer: int) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[layer][0, -1]  # (D,)
    return h.detach().cpu().to(torch.float32)

@torch.no_grad()
def paraphrase(model, tok, problem: str, max_new_tokens: int = 64) -> str:
    """Greedy one-shot paraphrase (keeps it deterministic)."""
    # Simple instruction; model-agnostic (works for non-chat).
    instr = (
        "Rephrase the following math problem concisely, preserving meaning and symbols. Do not add explanations or steps."
        f"{problem}\n\nParaphrase:"
    )
    enc = tok(instr, return_tensors="pt").to(model.device)
    gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens)
    out = tok.decode(gen[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
    # keep just the first line/sentence to avoid verbosity
    out = out.strip().split("\n")[0].strip()
    return out if out else problem

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="dir with math_eval.jsonl")
    p.add_argument("--data_path", required=True, help="data/MATH/train.jsonl (or test)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--out_dir", required=True, help="e.g., .../meaning_qonly")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--mode", choices=["prompt", "question_only", "paraphrase"], default="question_only")
    p.add_argument("--max_examples", type=int, default=None)
    args = p.parse_args()

    # load text data
    data = load_eval_pairs(args.data_dir, args.data_path)
    if args.max_examples:
        data = data[:args.max_examples]

    # load model/tokenizer once
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    # make dirs
    root = pathlib.Path(args.out_dir)
    (root / "vectors" / "meaning").mkdir(parents=True, exist_ok=True)

    for k, ex in enumerate(data):
        # choose the text whose last-token hidden state is the "meaning"
        if args.mode == "prompt":
            text = ex["prompt"].rstrip()
        elif args.mode == "paraphrase":
            para = paraphrase(model, tok, ex["problem"])
            text = f"Question: {para}"
        else:  # question_only
            text = f"Question: {ex['problem'].strip()}"

        for L in args.layers:
            tag = str(L)
            m_vec = hidden_last_token(model, tok, text, layer=L)  # (D,)
            qdir = root / "vectors" / "meaning" / str(k)
            qdir.mkdir(parents=True, exist_ok=True)
            np.save(qdir / f"m.l{tag}.npy", m_vec.numpy())

        if (k+1) % 50 == 0:
            print(f"[OK] wrote meanings for {k+1} examples")

    print(f"[DONE] saved question-only meanings -> {root}")

if __name__ == "__main__":
    main()
