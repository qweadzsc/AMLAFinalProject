"""Run LC evaluation on the three required validation settings."""

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
    parser = argparse.ArgumentParser(description="Evaluate LC/POMO on all Step 2 datasets.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "Project" / "baselines" / "lc_baseline" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=THIS_DIR / "results",
        help="Directory for per-dataset JSON metric files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, rel_data, node_cnt, pomo_size in DATASETS:
        output_json = args.results_dir / f"{name}.json"
        cmd = [
            sys.executable,
            str(THIS_DIR / "evaluate_lc_dataset.py"),
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
        ]
        subprocess.run(cmd, check=True)
        summary[name] = json.loads(output_json.read_text())

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
