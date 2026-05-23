"""Run ELG-lite evaluation on the three required validation settings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent


DATASETS = [
    ("tsp50_uniform", "Project/data/val/tsp50_uniform_val_128.txt", 50, 50),
    ("tsp50_ood", "Project/data/val/tsp50_ood_val_16.txt", 50, 50),
    ("tsp100_uniform", "Project/data/val/tsp100_uniform_val_16.txt", 100, 100),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one ELG-lite checkpoint on all Step 6 datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=THIS_DIR / "results",
        help="Directory for per-dataset JSON metric files.",
    )
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--dar-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dar-k", type=int, default=10)
    parser.add_argument("--dar-alpha", type=float, default=1.0)
    parser.add_argument("--dar-log-nearest", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, rel_data, node_cnt, pomo_size in DATASETS:
        output_json = args.results_dir / f"{name}.json"
        print(
            f"[eval_all_lc_elg] dataset={name} node_cnt={node_cnt} pomo_size={pomo_size} "
            f"checkpoint={args.checkpoint} dar_enabled={bool(args.dar_enabled)} "
            f"dar_k={args.dar_k} dar_alpha={args.dar_alpha}",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(THIS_DIR / "evaluate_lc_elg.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--test-data",
            str(REPO_ROOT / rel_data),
            "--node-cnt",
            str(node_cnt),
            "--pomo-size",
            str(pomo_size),
            "--device",
            args.device,
            "--output-json",
            str(output_json),
            "--dar-enabled",
            str(args.dar_enabled),
            "--dar-k",
            str(args.dar_k),
            "--dar-alpha",
            str(args.dar_alpha),
            "--dar-log-nearest",
            str(args.dar_log_nearest),
        ]
        if args.disable_progress:
            cmd.append("--disable-progress")
        subprocess.run(cmd, check=True)
        summary[name] = json.loads(output_json.read_text())
        print(
            f"[eval_all_lc_elg] finished dataset={name} avg_cost={summary[name]['avg_cost']:.4f} "
            f"avg_gap={summary[name].get('avg_gap', float('nan')):.2f}%",
            flush=True,
        )

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[eval_all_lc_elg] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
