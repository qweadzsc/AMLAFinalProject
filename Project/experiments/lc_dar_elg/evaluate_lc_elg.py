"""Evaluate an ELG-lite checkpoint on one TSP dataset."""

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from ml4co_kit import TSPEvaluator, TSPSolver
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
LC_BASELINE_DIR = REPO_ROOT / "Project" / "baselines" / "lc_baseline"
LOCAL_MODEL_DIR = THIS_DIR / "model"
sys.path.insert(0, str(LC_BASELINE_DIR))
sys.path.insert(0, str(LOCAL_MODEL_DIR))

from model import LCModel, TSPEnv  # noqa: E402
from local_policy import LocalPolicyScorer  # noqa: E402
from train_lc_elg import compute_ensemble_logits, prepare_first_step  # noqa: E402
from dar_wrapper import apply_dar_to_logits, logits_to_probs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one ELG-lite checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--node-cnt", type=int, required=True)
    parser.add_argument("--pomo-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--dar-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dar-k", type=int, default=10)
    parser.add_argument("--dar-alpha", type=float, default=1.0)
    parser.add_argument("--dar-log-nearest", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def load_solver(test_data: Path) -> TSPSolver:
    solver = TSPSolver()
    solver.from_txt(str(test_data), ref=True, normalize="uniform" not in str(test_data))
    return solver


def make_eval_args(checkpoint: dict, cli_args) -> Namespace:
    saved_args = checkpoint.get("args", {}).copy()
    saved_args.update(
        {
            "device": cli_args.device,
            "node_cnt": cli_args.node_cnt,
            "pomo_size": cli_args.pomo_size,
            "batch_size": 1,
            "disable_progress": cli_args.disable_progress,
            "dar_enabled": bool(cli_args.dar_enabled),
            "dar_k": cli_args.dar_k,
            "dar_alpha": cli_args.dar_alpha,
            "dar_log_nearest": cli_args.dar_log_nearest,
        }
    )
    return Namespace(**saved_args)


def load_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = LCModel(**checkpoint["model_params"])
    local_policy = LocalPolicyScorer(**checkpoint["local_policy_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    local_policy.load_state_dict(checkpoint["local_policy_state_dict"])
    model.to(device)
    local_policy.to(device)
    model.eval()
    local_policy.eval()
    return checkpoint, model, local_policy


def rollout(model, local_policy, env, eval_args):
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    state, reward, done = env.pre_step()
    while not done:
        if state.current_node is None:
            selected, _ = prepare_first_step(model, state, reset_state.coordinates)
        else:
            logits, _ = compute_ensemble_logits(model, local_policy, env, state, eval_args, epoch_idx=10**9)
            if eval_args.dar_enabled:
                logits = apply_dar_to_logits(
                    logits=logits,
                    coordinates=env.coordinates,
                    current_node=state.current_node,
                    ninf_mask=state.ninf_mask,
                    dar_k=eval_args.dar_k,
                    dar_alpha=eval_args.dar_alpha,
                    dar_log_nearest=bool(eval_args.dar_log_nearest),
                )
            selected = logits_to_probs(logits).argmax(dim=2)
        state, reward, done = env.step(selected)
    return reward


def evaluate(model, local_policy, env, solver, cli_args, eval_args):
    points = solver.points
    ref_tours = solver.ref_tours
    costs = []
    gaps = []
    ref_costs = []

    start_time = time.time()
    with torch.no_grad():
        iterator = tqdm(
            range(len(points)),
            desc="Evaluating ELG",
            unit="instance",
            disable=cli_args.disable_progress,
        )
        for i in iterator:
            coords = torch.from_numpy(points[i : i + 1]).float().to(cli_args.device)
            problems = torch.cdist(coords, coords, p=2)
            env.load_problems_manual(problems, coords)
            reward = rollout(model, local_policy, env, eval_args)
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
        "local_k": int(eval_args.local_k),
        "local_score_weight": float(eval_args.local_score_weight),
        "global_distance_penalty": float(eval_args.global_distance_penalty),
        "distance_k": int(eval_args.distance_k or eval_args.local_k),
        "dar_enabled": bool(eval_args.dar_enabled),
        "dar_k": int(eval_args.dar_k),
        "dar_alpha": float(eval_args.dar_alpha),
        "dar_log_nearest": bool(eval_args.dar_log_nearest),
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


def print_results(args, results):
    print("=" * 72)
    print("ELG-lite Evaluation")
    print("=" * 72)
    print(f"Dataset:             {args.test_data}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Node count:          {args.node_cnt}")
    print(f"POMO size:           {args.pomo_size}")
    print(f"Device:              {args.device}")
    print(f"Local K:             {results['local_k']}")
    print(f"Local weight:        {results['local_score_weight']}")
    print(f"Distance penalty:    {results['global_distance_penalty']}")
    print(f"DAR enabled:         {results['dar_enabled']}")
    print(f"DAR k:               {results['dar_k']}")
    print(f"DAR alpha:           {results['dar_alpha']}")
    print(f"DAR log nearest:     {results['dar_log_nearest']}")
    print(f"Average cost:        {results['avg_cost']:.4f}")
    if "avg_optimal_cost" in results:
        print(f"Average optimal:     {results['avg_optimal_cost']:.4f}")
    if "avg_gap" in results:
        print(f"Average gap:         {results['avg_gap']:.2f}%")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time/instance:   {results['avg_time_per_instance']:.4f}s")
    print("=" * 72)


def main():
    args = parse_args()
    if "cuda" in args.device:
        torch.cuda.set_device(int(args.device.split(":")[1]) if ":" in args.device else 0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    print(f"[evaluate_lc_elg] loading checkpoint={args.checkpoint}", flush=True)
    checkpoint, model, local_policy = load_checkpoint(args.checkpoint, args.device)
    eval_args = make_eval_args(checkpoint, args)
    solver = load_solver(args.test_data)
    env = TSPEnv(task="TSP", node_cnt=args.node_cnt, pomo_size=args.pomo_size)
    results = evaluate(model, local_policy, env, solver, args, eval_args)
    print_results(args, results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[evaluate_lc_elg] wrote results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
