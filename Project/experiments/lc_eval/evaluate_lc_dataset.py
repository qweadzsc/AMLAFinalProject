"""Standalone LC evaluator for fixed validation datasets.

This script intentionally lives outside Project/baselines/ so Step 2 can add
evaluation tooling without changing the original baseline implementation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ml4co_kit import TSPEvaluator, TSPSolver
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
LC_BASELINE_DIR = REPO_ROOT / "Project" / "baselines" / "lc_baseline"
sys.path.insert(0, str(LC_BASELINE_DIR))

from model import LCModel, TSPEnv  # noqa: E402


DEFAULT_MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "num_att_layers": 3,
    "qkv_dim": 16,
    "sqrt_qkv_dim": 16 ** 0.5,
    "num_heads": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LC/POMO on one TSP dataset.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=LC_BASELINE_DIR / "checkpoints" / "best_model.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="Path to a TSP txt file with reference tours.",
    )
    parser.add_argument("--node-cnt", type=int, required=True)
    parser.add_argument("--pomo-size", type=int, required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable metrics.",
    )
    return parser.parse_args()


def load_solver(test_data: Path) -> TSPSolver:
    solver = TSPSolver()
    solver.from_txt(str(test_data), ref=True, normalize="uniform" not in str(test_data))
    return solver


def make_model(checkpoint: Path, device: str) -> LCModel:
    model = LCModel(**DEFAULT_MODEL_PARAMS)
    state_dict = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate(model: LCModel, env: TSPEnv, solver: TSPSolver, device: str) -> dict:
    points = solver.points
    ref_tours = solver.ref_tours
    costs = []
    gaps = []
    ref_costs = []

    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(len(points)), desc="Evaluating", unit="instance"):
            coords = torch.from_numpy(points[i : i + 1]).float().to(device)
            problems = torch.cdist(coords, coords, p=2)
            env.load_problems_manual(problems, coords)

            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

            tour_length = -reward.max().item()
            costs.append(tour_length)

            if ref_tours is not None and len(ref_tours) > 0:
                ref_cost = TSPEvaluator(points[i]).evaluate(ref_tours[i])
                ref_costs.append(ref_cost)
                gaps.append((tour_length - ref_cost) / ref_cost * 100)

    total_time = time.time() - start_time
    results = {
        "num_instances": len(points),
        "avg_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "total_time": float(total_time),
        "avg_time_per_instance": float(total_time / len(points)),
    }
    if ref_costs:
        results.update(
            {
                "avg_optimal_cost": float(np.mean(ref_costs)),
                "std_optimal_cost": float(np.std(ref_costs)),
                "avg_gap": float(np.mean(gaps)),
                "std_gap": float(np.std(gaps)),
            }
        )
    return results


def print_results(args, results: dict) -> None:
    print("=" * 60)
    print("LC Evaluation")
    print("=" * 60)
    print(f"Dataset:             {args.test_data}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Node count:          {args.node_cnt}")
    print(f"POMO size:           {args.pomo_size}")
    print(f"Device:              {args.device}")
    print(f"Number of instances: {results['num_instances']}")
    print(f"Average cost:        {results['avg_cost']:.4f}")
    if "avg_optimal_cost" in results:
        print(f"Average optimal:     {results['avg_optimal_cost']:.4f}")
    if "avg_gap" in results:
        print(f"Average gap:         {results['avg_gap']:.2f}%")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time/instance:   {results['avg_time_per_instance']:.4f}s")
    print("=" * 60)


def main():
    args = parse_args()
    if "cuda" in args.device:
        torch.cuda.set_device(int(args.device.split(":")[1]) if ":" in args.device else 0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    solver = load_solver(args.test_data)
    model = make_model(args.checkpoint, args.device)
    env = TSPEnv(task="TSP", node_cnt=args.node_cnt, pomo_size=args.pomo_size)
    results = evaluate(model, env, solver, args.device)
    print_results(args, results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
