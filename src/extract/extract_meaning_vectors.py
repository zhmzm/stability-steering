# claim_1/extract_meaning_vectors.py
import os, json, argparse, pathlib, random
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def split_steps_by_para(response: str) -> List[str]:
    return [s for s in response.split("\n\n") if len(s.strip()) > 0]

def load_eval_pairs(data_dir: str, data_path: str) -> List[Dict]:
    lines = [json.loads(l) for l in open(data_path)]
    eval_lines = [json.loads(l) for l in open(os.path.join(data_dir, "math_eval.jsonl"))]
    lines = lines[:len(eval_lines)]
    out = []
    for d, e in zip(lines, eval_lines):
        out.append({
            "prompt": e["prompt"],
            "response": e["model_generation"][e["mv_index"]],
            "problem": d["problem"],
            "answer": e["answer"],
        })
    return out

def build_meaning_prefix(prompt_text: str, response_text: str, step_idx: int) -> str:
    steps = split_steps_by_para(response_text)
    prefix_steps = steps[:max(0, step_idx)]
    body = prompt_text.rstrip()
    if prefix_steps:
        if not body.endswith("\n"):
            body += "\n"
        body += "\n\n".join(prefix_steps)
    return body

@torch.no_grad()
def hidden_at_boundary(model, tok, prefix_text: str, layer: int) -> torch.Tensor:
    enc = tok(prefix_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**enc, output_hidden_states=True)
    return out.hidden_states[layer][0, -1].detach().cpu()

def unitize_if(v: torch.Tensor, do: bool) -> torch.Tensor:
    return l2norm(v.unsqueeze(0)).squeeze(0) if do else v

def main(args):
    random.seed(123)
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    hidden_path = os.path.join(args.hidden_dir, "hidden.pt")
    hidden = torch.load(hidden_path, map_location="cpu")
    data = load_eval_pairs(args.data_dir, args.data_path)

    Ls = args.layers if len(args.layers) > 0 else [args.layer]
    root = pathlib.Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
    vec_root = root / "vectors"
    (vec_root / "meaning").mkdir(parents=True, exist_ok=True)
    res_root = root / "results" / "exp1_negative_pairing" / "per_example"
    res_root.mkdir(parents=True, exist_ok=True)

    # choose which indices to use
    mode = args.mode.lower()  # 'r' or 't'
    assert mode in ("r","t")
    plus_dirname = "rplus" if mode == "r" else "tplus"
    pair_dirname = "rpair" if mode == "r" else "tpair"
    mean_stub   = f"mean_{pair_dirname}"  # mean_rpair / mean_tpair

    pairs_per_layer = {L: [] for L in Ls}

    K_diff = args.k_diff
    for k, ex in enumerate(data):
        meta = hidden[0][k]
        check_idx: torch.Tensor = meta["check_index"]
        switch_idx: torch.Tensor = meta["switch_index"]

        # pick boundary
        idx_tensor = check_idx if mode == "r" else switch_idx
        if idx_tensor.numel() == 0:
            continue
        b_idx = int(idx_tensor[0].item())

        prefix_text = build_meaning_prefix(ex["prompt"], ex["response"], b_idx)

        per_layer_rows = []
        for L in Ls:
            tag = f"l{L if L >= 0 else 'last'}"
            h_steps = hidden[L][k]["step"]
            if b_idx >= h_steps.shape[0]:
                continue
            h_plus = h_steps[b_idx]  # R+ or T+

            h_plus_for_pair = h_plus.clone()
            vm = hidden_at_boundary(model, tok, prefix_text, layer=L)
            vm_for_pair = vm.clone()

            # buckets for cos diagnostics (unchanged)
            def gather(ids):
                if len(ids) == 0: return None
                X = h_steps[ids].clone()
                X = unitize_if(X, args.unit_norm) if X.ndim == 1 else (X / (X.norm(dim=-1, keepdim=True)+1e-12)) if args.unit_norm else X
                mu = X.mean(dim=0)
                return unitize_if(mu, args.unit_norm)

            R_set = set(hidden[L][k]["check_index"].tolist())
            T_set = set(hidden[L][k]["switch_index"].tolist())
            S_ids = set(range(h_steps.shape[0]))
            E_ids = sorted(list(S_ids - R_set - T_set))
            T_ids = sorted(list(T_set))
            E_same = gather(E_ids)
            T_same = gather(T_ids)

            def sample_diff(label_fn):
                vecs = []
                for kk in range(len(data)):
                    if kk == k: 
                        continue
                    meta_k = hidden[L][kk]
                    S_k = meta_k["step"]
                    ids_k = label_fn(meta_k)
                    if len(ids_k) == 0: 
                        continue
                    import random as _rnd
                    pick = _rnd.sample(ids_k, k=min(K_diff, len(ids_k)))
                    V = S_k[pick]
                    V = (V / (V.norm(dim=-1, keepdim=True)+1e-12)) if args.unit_norm else V
                    vecs.append(V)
                if not vecs:
                    return None
                X = torch.cat(vecs, dim=0)
                mu = X.mean(dim=0)
                return unitize_if(mu, args.unit_norm)

            E_diff = sample_diff(lambda m: sorted(list(
                set(range(m["step"].shape[0])) - set(m["check_index"].tolist()) - set(m["switch_index"].tolist())
            )))
            T_diff = sample_diff(lambda m: m["switch_index"].tolist())

            def cos(a,b):
                if a is None or b is None: return None
                return torch.nn.functional.cosine_similarity(a.view(1,-1), b.view(1,-1)).item()

            h_plus_cos = unitize_if(h_plus, args.unit_norm)
            vm_cos     = unitize_if(vm,     args.unit_norm)
            row = {
                "qid": k, "layer": L, "b_step_idx": b_idx,
                "cos_meaning": cos(h_plus_cos, vm_cos),
                "cos_E_same":  cos(h_plus_cos, E_same),
                "cos_T_same":  cos(h_plus_cos, T_same),
                "cos_E_diff":  cos(h_plus_cos, E_diff),
                "cos_T_diff":  cos(h_plus_cos, T_diff),
            }
            per_layer_rows.append(row)

            # saves (mode-aware)
            qdir = vec_root / "meaning" / str(k)
            qdir.mkdir(parents=True, exist_ok=True)
            np.save(qdir / f"m.{tag}.npy", vm_for_pair.to(torch.float32).numpy())

            pdir_plus = vec_root / plus_dirname / str(k)
            pdir_plus.mkdir(parents=True, exist_ok=True)
            np.save(pdir_plus / f"{'r' if mode=='r' else 't'}.{tag}.npy", h_plus_for_pair.to(torch.float32).numpy())

            pdir_pair = vec_root / pair_dirname / str(k)
            pdir_pair.mkdir(parents=True, exist_ok=True)
            pair_vec = (h_plus_for_pair - vm_for_pair).to(torch.float32).numpy()
            np.save(pdir_pair / f"pair.{tag}.npy", pair_vec)

            pairs_per_layer[L].append(pair_vec)

        if per_layer_rows:
            out_csv = res_root / f"{k}_layers.csv"
            import csv
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=per_layer_rows[0].keys())
                w.writeheader()
                for r in per_layer_rows:
                    w.writerow(r)

    # save layer means for steering (mode-aware)
    for L in Ls:
        tag = f"l{L if L >= 0 else 'last'}"
        pairs = pairs_per_layer[L]
        if not pairs: 
            continue
        P = np.vstack(pairs).astype(np.float32)
        mean_pair = P.mean(0).astype(np.float32)
        mean_path = vec_root / f"{mean_stub}.{tag}.pt"
        unit_path = vec_root / f"{mean_stub}.{tag}.unit.pt"
        torch.save(torch.from_numpy(mean_pair), mean_path)
        unit = mean_pair / (np.linalg.norm(mean_pair) + 1e-12)
        torch.save(torch.from_numpy(unit.astype(np.float32)), unit_path)
        print(f"[SAVED] {mean_path} and unit -> {unit_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--hidden_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_dir", default="output_exp1")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--layers", type=int, nargs="*", default=[20])
    ap.add_argument("--k_diff", type=int, default=20)
    ap.add_argument("--unit_norm", action="store_true", default=True)
    ap.add_argument("--mode", choices=["r","t"], required=True,
                    help="r=Reflection boundary (check_index), t=Transition boundary (switch_index)")
    args = ap.parse_args()
    main(args)
