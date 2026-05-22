"""Grid search DAR hyperparameters on the three required validation settings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent

DATASET_ORDER = ["tsp50_uniform", "tsp50_ood", "tsp100_uniform"]


def parse_csv_ints(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep DAR hyperparameters.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to evaluate.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dar-log-nearest", type=int, choices=[0, 1], default=1)
    parser.add_argument("--sweep-k", default="5,10,20,50")
    parser.add_argument("--sweep-alpha", default="0.25,0.5,1.0,2.0")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory for all sweep outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ks = parse_csv_ints(args.sweep_k)
    alphas = parse_csv_floats(args.sweep_alpha)

    sweep_results = {}

    for k in ks:
        for alpha in alphas:
            run_name = f"k{k}_a{str(alpha).replace('.', 'p')}"
            run_dir = args.results_dir / run_name
            cmd = [
                sys.executable,
                str(THIS_DIR / "eval_all_lc_dar.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                args.device,
                "--dar-enabled",
                "1",
                "--dar-k",
                str(k),
                "--dar-alpha",
                str(alpha),
                "--dar-log-nearest",
                str(args.dar_log_nearest),
                "--results-dir",
                str(run_dir),
            ]
            subprocess.run(cmd, check=True)
            sweep_results[run_name] = json.loads((run_dir / "summary.json").read_text())

    summary_path = args.results_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(sweep_results, indent=2) + "\n")

    lines = [
        "# Step 3 DAR Sweep",
        "",
        "| Run | K | Alpha | TSP50 Uniform Gap | TSP50 OOD Gap | TSP100 Gap | Mean Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    best_by_dataset = {name: None for name in DATASET_ORDER}
    best_mean = None

    for run_name, result in sweep_results.items():
        k_str, alpha_str = run_name.split("_")
        k = int(k_str[1:])
        alpha = float(alpha_str[1:].replace("p", "."))
        gaps = [result[name]["avg_gap"] for name in DATASET_ORDER]
        mean_gap = sum(gaps) / len(gaps)
        lines.append(
            f"| {run_name} | {k} | {alpha:.2f} | {gaps[0]:.2f}% | {gaps[1]:.2f}% | {gaps[2]:.2f}% | {mean_gap:.2f}% |"
        )

        if best_mean is None or mean_gap < best_mean["mean_gap"]:
            best_mean = {"run_name": run_name, "k": k, "alpha": alpha, "mean_gap": mean_gap}

        for dataset_name, gap in zip(DATASET_ORDER, gaps):
            if best_by_dataset[dataset_name] is None or gap < best_by_dataset[dataset_name]["gap"]:
                best_by_dataset[dataset_name] = {"run_name": run_name, "k": k, "alpha": alpha, "gap": gap}

    lines.extend(
        [
            "",
            "## Best By Dataset",
            "",
            "| Dataset | Run | K | Alpha | Gap |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for dataset_name in DATASET_ORDER:
        best = best_by_dataset[dataset_name]
        lines.append(f"| {dataset_name} | {best['run_name']} | {best['k']} | {best['alpha']:.2f} | {best['gap']:.2f}% |")

    lines.extend(
        [
            "",
            "## Best Mean Gap",
            "",
            f"- Run: `{best_mean['run_name']}`",
            f"- K: `{best_mean['k']}`",
            f"- Alpha: `{best_mean['alpha']:.2f}`",
            f"- Mean gap: `{best_mean['mean_gap']:.2f}%`",
        ]
    )

    comparison_path = args.results_dir / "comparison.md"
    comparison_path.write_text("\n".join(lines) + "\n")

    best_path = args.results_dir / "best_summary.json"
    best_path.write_text(
        json.dumps({"best_by_dataset": best_by_dataset, "best_mean": best_mean}, indent=2) + "\n"
    )

    print(f"Wrote sweep summary to {summary_path}")
    print(f"Wrote comparison table to {comparison_path}")


if __name__ == "__main__":
    main()
