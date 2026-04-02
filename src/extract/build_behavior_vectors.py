#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer


CHECK_WORDS = [
    "verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect",
    "let me check", "seems right", "re-check", "double check", "double-check",
    "check again", "reconsider", "sanity check", "confirm", "validate",
]
CHECK_PREFIX = ["wait", "hmm", "let's check", "let me check"]

SWITCH_WORDS = [
    "another way", "another approach", "different approach", "another method",
    "another solution", "another strategy", "try a different", "change approach", "switch",
]
SWITCH_PREFIX = ["alternatively"]


def paragraph_steps(text: str):
    t = text.replace("\r\n", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return [s.strip() for s in t.split("\n\n") if s.strip()]


def looks_like_r(paragraph: str) -> bool:
    s = paragraph.strip().lower()
    return any(s.startswith(p) for p in CHECK_PREFIX) or any(w in s for w in CHECK_WORDS)


def looks_like_t(paragraph: str) -> bool:
    s = paragraph.strip().lower()
    return any(s.startswith(p) for p in SWITCH_PREFIX) or any(w in s for w in SWITCH_WORDS)


def looks_naturally_finished(text: str, tok: AutoTokenizer, max_new_tokens: int) -> bool:
    s = (text or "").strip()
    if "\\boxed" not in s:
        return False
    if len(tok.encode(s, add_special_tokens=False)) >= max_new_tokens:
        return False
    if s[-1] not in "}.!?]$)":
        return False
    tail = s[-80:].lower()
    if re.search(r"(=|\\frac|\\sqrt|\\left|\\right|\\cup|\\cap|\\in)\s*$", tail):
        return False
    return True


def response_text(row: dict) -> str:
    mg = row.get("model_generation", "")
    if isinstance(mg, list):
        idx = int(row.get("mv_index", 0))
        if 0 <= idx < len(mg):
            return mg[idx]
        return mg[0] if mg else ""
    return mg


def build_execution_vectors(hidden, layer: int):
    rows, ex_ids = [], []
    layer_dict = hidden[layer]
    for k in sorted(layer_dict.keys()):
        step = layer_dict[k]["step"]
        r_ids = set(layer_dict[k]["check_index"].tolist())
        t_ids = set(layer_dict[k]["switch_index"].tolist())
        e_ids = sorted(set(range(step.shape[0])) - r_ids - t_ids)
        non_e = sorted(r_ids | t_ids)
        if len(e_ids) == 0 or len(non_e) == 0:
            continue
        pos = step[e_ids].mean(0)
        neg = step[non_e].mean(0)
        rows.append((pos - neg).unsqueeze(0))
        ex_ids.append(str(k))
    return rows, ex_ids


def build_ending_vectors(hidden, rows_jsonl, layer: int, tok: AutoTokenizer, max_new_tokens: int):
    rows, ex_ids = [], []
    layer_dict = hidden[layer]
    for k in sorted(layer_dict.keys()):
        step = layer_dict[k]["step"]
        if step.shape[0] < 2:
            continue
        resp = response_text(rows_jsonl[k])
        if not looks_naturally_finished(resp, tok, max_new_tokens):
            continue
        pos = step[-1]
        neg = step[:-1].mean(0)
        rows.append((pos - neg).unsqueeze(0))
        ex_ids.append(str(k))
    return rows, ex_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--hidden_subdir", default="hidden_mv")
    ap.add_argument("--layers", type=int, nargs="+", default=[20])
    ap.add_argument("--save_prefix", required=True)
    ap.add_argument("--mode", choices=["e", "end"], required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--tokenizer_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    hidden = torch.load(data_dir / args.hidden_subdir / "hidden.pt", weights_only=False)
    save_dir = data_dir / f"vector_{args.save_prefix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    rows_jsonl = None
    tok = None
    if args.mode == "end":
        with open(data_dir / "math_eval.jsonl") as f:
            rows_jsonl = [json.loads(line) for line in f]
        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)

    for layer in args.layers:
        if args.mode == "e":
            rows, ex_ids = build_execution_vectors(hidden, layer)
        else:
            rows, ex_ids = build_ending_vectors(hidden, rows_jsonl, layer, tok, args.max_new_tokens)
        if not rows:
            print(f"[warn] no rows for mode={args.mode} layer={layer}")
            continue
        mat = torch.cat(rows, dim=0)
        vec_path = save_dir / f"layer_{layer}_transition_reflection_steervec.pt"
        man_path = save_dir / f"layer_{layer}_examples.json"
        if args.overwrite or not vec_path.exists():
            torch.save(mat, vec_path)
            man_path.write_text(json.dumps(ex_ids) + "\n")
            print(f"[OK] L{layer}: saved {tuple(mat.shape)} to {vec_path}")
        else:
            print(f"[skip] {vec_path}")


if __name__ == "__main__":
    main()
