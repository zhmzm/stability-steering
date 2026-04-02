#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_acc(path: Path):
    try:
        return json.loads(path.read_text()).get("acc")
    except Exception:
        return None


def find_baseline_metrics(root: Path) -> Path | None:
    matches = sorted(root.glob("baseline/base_run/**/metrics.json"))
    return matches[0] if matches else None


def collect_condition(root: Path, condition: str):
    rows = []
    for metrics_fp in sorted((root / condition).glob("**/coef_*/base_remove_bos/metrics.json")):
        coef = metrics_fp.parent.parent.name.replace("coef_", "")
        acc = load_acc(metrics_fp)
        if acc is None:
            continue
        rows.append({
            "condition": condition,
            "coef": coef,
            "acc": acc,
            "metrics_path": str(metrics_fp),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_best_json", required=True)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    baseline_fp = find_baseline_metrics(eval_root)
    baseline_acc = load_acc(baseline_fp) if baseline_fp else None

    all_rows = []
    best = {
        "baseline": {
            "acc": baseline_acc,
            "metrics_path": str(baseline_fp) if baseline_fp else None,
        }
    }

    for condition in ("claimSeal", "claimA", "claimB", "claimAB"):
        rows = collect_condition(eval_root, condition)
        all_rows.extend(rows)
        if rows:
            rows_sorted = sorted(rows, key=lambda r: r["acc"], reverse=True)
            top = rows_sorted[0]
            best[condition] = {
                "best_coef": top["coef"],
                "best_acc": top["acc"],
                "delta_vs_baseline": None if baseline_acc is None else top["acc"] - baseline_acc,
                "metrics_path": top["metrics_path"],
            }
        else:
            best[condition] = None

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=["condition", "coef", "acc", "metrics_path"])
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    out_best = Path(args.out_best_json)
    out_best.parent.mkdir(parents=True, exist_ok=True)
    out_best.write_text(json.dumps(best, indent=2) + "\n")

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_best}")


if __name__ == "__main__":
    main()
