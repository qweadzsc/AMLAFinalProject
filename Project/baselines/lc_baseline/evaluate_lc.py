"""LC final evaluation script with ELG-lite + tuned DAR."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from ml4co_kit import TSPEvaluator, TSPSolver
from tqdm import tqdm

from model import LCModel, TSPEnv


THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "checkpoints" / "best_model.pth"
TEST_DATA_PATH = THIS_DIR.parent.parent / "data" / "val" / "tsp50_uniform_val_128.txt"
DEVICE = os.environ.get("AMLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 128
NUM_ATT_LAYERS = 3
NUM_HEADS = 8
QKV_DIM = 16
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10
LOCAL_POLICY_DIM = 128
LOCAL_K = 10
LOCAL_SCORE_WEIGHT = 1.0
GLOBAL_DISTANCE_PENALTY = 0.5
DISTANCE_K = 10
DAR_ENABLED = 1
DAR_K = 20
DAR_ALPHA = 0.5
DAR_LOG_NEAREST = 1
NODE_CNT = 50
POMO_SIZE = 50

model_params = {
    "embedding_dim": EMBEDDING_DIM,
    "sqrt_embedding_dim": EMBEDDING_DIM ** 0.5,
    "num_att_layers": NUM_ATT_LAYERS,
    "qkv_dim": QKV_DIM,
    "sqrt_qkv_dim": QKV_DIM ** 0.5,
    "num_heads": NUM_HEADS,
    "logit_clipping": LOGIT_CLIPPING,
    "ff_hidden_dim": FF_HIDDEN_DIM,
    "eval_type": "argmax",
    "local_policy_dim": LOCAL_POLICY_DIM,
    "local_k": LOCAL_K,
    "local_score_weight": LOCAL_SCORE_WEIGHT,
    "global_distance_penalty": GLOBAL_DISTANCE_PENALTY,
    "distance_k": DISTANCE_K,
    "dar_enabled": DAR_ENABLED,
    "dar_k": DAR_K,
    "dar_alpha": DAR_ALPHA,
    "dar_log_nearest": DAR_LOG_NEAREST,
    "max_positional_rank": 128,
}

env_params = {
    "task": "TSP",
    "node_cnt": NODE_CNT,
    "pomo_size": POMO_SIZE,
}


def evaluate_model(model, env, test_solver, device):
    model.eval()
    test_points = test_solver.points
    test_ref_tours = test_solver.ref_tours
    num_instances = len(test_points)
    costs = []
    gaps = []
    ref_costs = []

    print(f"\nEvaluating on {num_instances} instances...")
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(num_instances), desc="Evaluating", unit="instance"):
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
        "num_instances": num_instances,
        "avg_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "total_time": float(total_time),
        "avg_time_per_instance": float(total_time / num_instances),
    }
    if ref_costs:
        results["avg_optimal_cost"] = float(np.mean(ref_costs))
        results["std_optimal_cost"] = float(np.std(ref_costs))
        results["avg_gap"] = float(np.mean(gaps))
        results["std_gap"] = float(np.std(gaps))
    return results


def main():
    print("=" * 72)
    print("LC Final Evaluation - ELG-lite + tuned DAR")
    print("=" * 72)
    if "cuda" in DEVICE:
        device_index = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
        torch.cuda.set_device(device_index)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    print(f"Loading test data from {TEST_DATA_PATH}...")
    solver = TSPSolver()
    solver.from_txt(str(TEST_DATA_PATH), ref=True, normalize="uniform" not in str(TEST_DATA_PATH))
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
    print(f"Number of instances: {results['num_instances']}")
    print(f"Average cost:        {results['avg_cost']:.4f}")
    if "avg_optimal_cost" in results:
        print(f"Average optimal:     {results['avg_optimal_cost']:.4f}")
    if "avg_gap" in results:
        print(f"Average gap:         {results['avg_gap']:.2f}%")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time/instance:   {results['avg_time_per_instance']:.4f}s")
    print("=" * 72)
