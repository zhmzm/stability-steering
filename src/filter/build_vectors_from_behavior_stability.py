#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_matrix_and_manifest(vec_dir: Path, vec_file: str, layer: int):
    vp = vec_dir / vec_file.format(layer=layer)
    mp = vec_dir / f"layer_{layer}_examples.json"
    v = torch.load(vp, map_location="cpu")
    if hasattr(v, "numpy"):
        v = v.numpy()
    ex_ids = json.loads(mp.read_text())
    return v.astype(np.float32), [int(x) for x in ex_ids]


def unit(x: np.ndarray) -> np.ndarray:
    return (x / (float(np.linalg.norm(x)) + 1e-12)).astype(np.float32, copy=False)


def pick_indices(per_boundary_csv: Path, ex_ids: list[int], want: str):
    stable_q, unstable_q = set(), set()
    with open(per_boundary_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = int(row["qid"])
            if int(row["is_stable"]) == 1:
                stable_q.add(qid)
            else:
                unstable_q.add(qid)
    if want == "stable":
        return [i for i, q in enumerate(ex_ids) if q in stable_q]
    if want == "unstable":
        return [i for i, q in enumerate(ex_ids) if q in unstable_q]
    return list(range(len(ex_ids)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector_dir", required=True)
    ap.add_argument("--vector_file", required=True, help="format string, e.g. layer_{layer}_execution.pt")
    ap.add_argument("--per_boundary_csv", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    V, ex_ids = load_matrix_and_manifest(Path(args.vector_dir), args.vector_file, args.layer)

    for label in ("seal", "stable", "unstable"):
        keep = pick_indices(Path(args.per_boundary_csv), ex_ids, "all" if label == "seal" else label)
        if not keep:
            print(f"[warn] no examples for {label}, skipping")
            continue
        mean = V[keep].mean(axis=0).astype(np.float32)
        pt = out / f"{label}.pt"
        upt = out / f"{label}.unit.pt"
        if not args.overwrite and pt.exists() and upt.exists():
            print(f"[skip] {label}")
            continue
        torch.save(torch.from_numpy(mean.copy()), pt)
        torch.save(torch.from_numpy(unit(mean)), upt)
        print(f"[OK] wrote {label}: {pt} / {upt} (N={len(keep)})")


if __name__ == "__main__":
    main()
