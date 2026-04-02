#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def unit(t: torch.Tensor) -> torch.Tensor:
    return t / (t.norm() + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec_a", required=True)
    ap.add_argument("--vec_b", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default="claimAB")
    args = ap.parse_args()

    va = torch.load(args.vec_a, map_location="cpu")
    vb = torch.load(args.vec_b, map_location="cpu")
    combo = va.to(torch.float32) + vb.to(torch.float32)
    combo_unit = unit(combo)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{args.name}.l20.pt"
    unit_path = out_dir / f"{args.name}.l20.unit.pt"
    meta_path = out_dir / f"{args.name}.meta.json"

    torch.save(combo, raw_path)
    torch.save(combo_unit, unit_path)
    meta_path.write_text(json.dumps({
        "vec_a": args.vec_a,
        "vec_b": args.vec_b,
        "raw_out": str(raw_path),
        "unit_out": str(unit_path),
    }, indent=2) + "\n")

    print(f"[OK] wrote {raw_path}")
    print(f"[OK] wrote {unit_path}")
    print(f"[OK] wrote {meta_path}")


if __name__ == "__main__":
    main()
