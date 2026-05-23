"""Final LC evaluation script with ELG-lite + tuned DAR."""

import time
from pathlib import Path

import numpy as np
import torch
from ml4co_kit import TSPEvaluator, TSPSolver
from tqdm import tqdm

from model import LCModel, TSPEnv


THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "checkpoints" / "best_model.pth"
TEST_DATA_PATH = THIS_DIR.parent / "data" / "val" / "tsp50_uniform_val_128.txt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model_params = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "num_att_layers": 3,
    "qkv_dim": 16,
    "sqrt_qkv_dim": 16 ** 0.5,
    "num_heads": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
    "local_policy_dim": 128,
    "local_k": 10,
    "local_score_weight": 1.0,
    "global_distance_penalty": 0.5,
    "distance_k": 10,
    "dar_enabled": 1,
    "dar_k": 20,
    "dar_alpha": 0.5,
    "dar_log_nearest": 1,
    "max_positional_rank": 128,
}

env_params = {
    "task": "TSP",
    "node_cnt": 50,
    "pomo_size": 50,
}


def evaluate_model(model, env, test_solver, device):
    model.eval()
    test_points = test_solver.points
    test_ref_tours = test_solver.ref_tours
    costs = []
    gaps = []
    ref_costs = []

    print(f"\nEvaluating on {len(test_points)} instances...")
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(len(test_points)), desc="Evaluating", unit="instance"):
            coords = torch.from_numpy(test_points[i:i + 1]).float().to(device)
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
            if test_ref_tours is not None and len(test_ref_tours) > 0:
                ref_cost = TSPEvaluator(test_points[i]).evaluate(test_ref_tours[i])
                ref_costs.append(ref_cost)
                gaps.append((tour_length - ref_cost) / ref_cost * 100)

    total_time = time.time() - start_time
    results = {
        "num_instances": len(test_points),
        "avg_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "total_time": float(total_time),
        "avg_time_per_instance": float(total_time / len(test_points)),
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


def main():
    print("=" * 72)
    print("Final LC Evaluation - ELG-lite + tuned DAR")
    print("=" * 72)
    if "cuda" in DEVICE:
        device_index = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
        torch.cuda.set_device(device_index)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    print(f"Loading test data from {TEST_DATA_PATH}...")
    solver = TSPSolver()
    solver.from_txt(str(TEST_DATA_PATH), ref=True)
    print(f"Loaded {len(solver.points)} test instances")

    print(f"Loading model from {MODEL_PATH}...")
    model = LCModel(**model_params)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE if "cuda" not in DEVICE else None)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    env = TSPEnv(**env_params)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params:,} parameters")
    results = evaluate_model(model, env, solver, DEVICE)

    print("\n" + "=" * 72)
    print("Evaluation Results")
    print("=" * 72)
    print(f"Average cost:        {results['avg_cost']:.4f}")
    if "avg_optimal_cost" in results:
        print(f"Average optimal:     {results['avg_optimal_cost']:.4f}")
    if "avg_gap" in results:
        print(f"Average gap:         {results['avg_gap']:.2f}%")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time/instance:   {results['avg_time_per_instance']:.4f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
