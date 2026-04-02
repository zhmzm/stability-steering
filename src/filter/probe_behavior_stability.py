#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def looks_like_e(paragraph: str) -> bool:
    return not looks_like_r(paragraph) and not looks_like_t(paragraph)


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


def load_eval_pairs(data_dir: str, data_path: str):
    lines = [json.loads(l) for l in open(data_path)]
    eval_lines = [json.loads(l) for l in open(os.path.join(data_dir, "math_eval.jsonl"))]
    lines = lines[:len(eval_lines)]
    out = []
    for d, e in zip(lines, eval_lines):
        mv_idx = int(e.get("mv_index", 0))
        resp = e["model_generation"][mv_idx]
        out.append({
            "prompt": e["prompt"],
            "response": resp,
            "problem": d["problem"],
            "answer": d.get("answer", e.get("answer")),
        })
    return out


def build_prefix(prompt_text: str, response_text: str, step_idx: int) -> str:
    steps = paragraph_steps(response_text)
    prefix_steps = steps[:max(0, step_idx)]
    body = prompt_text.rstrip()
    if prefix_steps:
        if not body.endswith("\n"):
            body += "\n"
        body += "\n\n".join(prefix_steps)
    return body


@torch.no_grad()
def sample_continuation(model, tok, prefix: str, max_new_tokens: int, seed: int) -> str:
    enc = tok(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.random.fork_rng(devices=[model.device]):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def boundary_indices(meta_q: dict, mode: str):
    if mode == "r":
        return meta_q["check_index"].tolist()
    if mode == "t":
        return meta_q["switch_index"].tolist()
    if mode == "e":
        r_ids = set(meta_q["check_index"].tolist())
        t_ids = set(meta_q["switch_index"].tolist())
        return sorted(set(range(meta_q["step"].shape[0])) - r_ids - t_ids)
    if mode == "end":
        return [int(meta_q["step"].shape[0] - 1)] if meta_q["step"].shape[0] >= 2 else []
    raise ValueError(mode)


def is_target_event(text: str, tok: AutoTokenizer, mode: str, max_new_tokens: int) -> bool:
    if mode == "r":
        paras = paragraph_steps(text)
        return bool(paras) and looks_like_r(paras[0])
    if mode == "t":
        paras = paragraph_steps(text)
        return bool(paras) and looks_like_t(paras[0])
    if mode == "e":
        paras = paragraph_steps(text)
        return bool(paras) and looks_like_e(paras[0])
    if mode == "end":
        return looks_naturally_finished(text, tok, max_new_tokens)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--hidden_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["r", "t", "e", "end"], required=True)
    ap.add_argument("--qid_start", type=int, default=None)
    ap.add_argument("--qid_end", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    data = load_eval_pairs(args.data_dir, args.data_path)
    H = torch.load(os.path.join(args.hidden_dir, "hidden.pt"), map_location="cpu")
    meta0 = H[0]

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    qids = list(range(len(data)))
    if args.qid_start is not None and args.qid_end is not None:
        qids = list(range(max(0, args.qid_start), min(len(data), args.qid_end)))

    total_bounds = total_stable = 0
    for qid in qids:
        idxs = boundary_indices(meta0[qid], args.mode)
        bd_fp = parts_dir / f"per_boundary.qid={qid}.csv"
        ex_fp = parts_dir / f"per_example.qid={qid}.csv"
        n_stable = 0
        with open(str(bd_fp) + ".tmp", "w", newline="") as bf:
            bw = csv.DictWriter(bf, fieldnames=["qid", "layer", "boundary_idx", "step_idx", "n_evt", "is_stable", "evt"])
            bw.writeheader()
            for b_idx, step_idx in enumerate(idxs):
                prefix = build_prefix(data[qid]["prompt"], data[qid]["response"], int(step_idx))
                n_evt = 0
                for s in range(args.n_samples):
                    seed = 3000 + 37 * qid + 53 * b_idx + s
                    text = sample_continuation(model, tok, prefix, args.max_new_tokens, seed)
                    if is_target_event(text, tok, args.mode, args.max_new_tokens):
                        n_evt += 1
                is_stable = int(n_evt >= 8)
                n_stable += is_stable
                bw.writerow({
                    "qid": qid,
                    "layer": args.layer,
                    "boundary_idx": b_idx,
                    "step_idx": int(step_idx),
                    "n_evt": n_evt,
                    "is_stable": is_stable,
                    "evt": args.mode,
                })
        os.replace(str(bd_fp) + ".tmp", bd_fp)
        with open(str(ex_fp) + ".tmp", "w", newline="") as ef:
            ew = csv.DictWriter(ef, fieldnames=["qid", "layer", "n_boundaries", "n_stable", "n_unstable", "evt"])
            ew.writeheader()
            ew.writerow({
                "qid": qid,
                "layer": args.layer,
                "n_boundaries": len(idxs),
                "n_stable": n_stable,
                "n_unstable": len(idxs) - n_stable,
                "evt": args.mode,
            })
        os.replace(str(ex_fp) + ".tmp", ex_fp)
        total_bounds += len(idxs)
        total_stable += n_stable

    with open(parts_dir / f"summary.{args.mode}.qids={qids[0]}-{qids[-1]}.json", "w") as f:
        json.dump({
            "qids": qids,
            "layer": args.layer,
            "evt": args.mode,
            "n_boundaries_total": total_bounds,
            "n_stable_total": total_stable,
            "n_unstable_total": total_bounds - total_stable,
            "rule": "Stable if n_evt >= 8 (out of 10); else Unstable.",
        }, f, indent=2)
    print(f"[OK] [{args.mode}] wrote parts in: {parts_dir}")


if __name__ == "__main__":
    main()
