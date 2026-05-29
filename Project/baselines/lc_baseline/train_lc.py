"""LC final training script with ELG-lite local policy."""

import os
import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ml4co_kit import TSPSolver
from tqdm import tqdm

from model import LCModel, TSPEnv


THIS_DIR = Path(__file__).resolve().parent
VAL_DATA_PATH = THIS_DIR.parent.parent / "data" / "val" / "tsp50_uniform_val_128.txt"
SAVE_DIR = THIS_DIR / "checkpoints"


def build_parser():
    parser = ArgumentParser(description="Train the final LC model with ELG-lite style local policy.")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("AMLA_EPOCHS", 20)))
    parser.add_argument("--batches-per-epoch", type=int, default=int(os.environ.get("AMLA_BATCHES_PER_EPOCH", 50)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("AMLA_BATCH_SIZE", 64)))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("AMLA_LEARNING_RATE", 1e-4)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AMLA_SEED", 20260522)))
    parser.add_argument("--val-interval", type=int, default=int(os.environ.get("AMLA_VAL_INTERVAL", 5)))
    parser.add_argument("--device", default=os.environ.get("AMLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--node-cnt", type=int, default=50)
    parser.add_argument("--pomo-size", type=int, default=50)
    parser.add_argument("--local-policy-dim", type=int, default=128)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--local-score-weight", type=float, default=1.0)
    parser.add_argument("--global-distance-penalty", type=float, default=0.5)
    parser.add_argument("--distance-k", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--val-data", type=Path, default=VAL_DATA_PATH)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(device: str) -> None:
    if "cuda" in device:
        torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")


def train_one_batch(model, env, optimizer, batch_size):
    env.load_problems(batch_size)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)
    state, reward, done = env.pre_step()
    prob_list = []

    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        if prob is not None:
            prob_list.append(prob)

    advantage = reward - reward.mean(dim=1, keepdim=True)
    log_prob = torch.stack(prob_list, dim=2).log().sum(dim=2)
    loss = -(advantage.detach() * log_prob).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "avg_length": -reward.mean().item(),
        "best_length": -reward.max(dim=1).values.mean().item(),
        "loss": loss.item(),
    }


def validate(model, env, val_solver, device) -> float:
    model.eval()
    total_cost = 0.0
    print("[train_lc] starting validation", flush=True)
    with torch.no_grad():
        for i in tqdm(range(len(val_solver.points)), desc="Validate", unit="instance", leave=False):
            coords = torch.from_numpy(val_solver.points[i:i + 1]).float().to(device)
            problems = torch.cdist(coords, coords, p=2)
            env.load_problems_manual(problems, coords)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
            total_cost += -reward.max().item()
    model.train()
    return total_cost / len(val_solver.points)


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    configure_device(args.device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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
        "local_policy_dim": args.local_policy_dim,
        "local_k": args.local_k,
        "local_score_weight": args.local_score_weight,
        "global_distance_penalty": args.global_distance_penalty,
        "distance_k": args.distance_k,
        "dar_enabled": 0,
        "dar_k": 20,
        "dar_alpha": 0.5,
        "dar_log_nearest": 1,
        "max_positional_rank": max(128, args.node_cnt),
    }
    env_params = {"task": "TSP", "node_cnt": args.node_cnt, "pomo_size": args.pomo_size}

    model = LCModel(**model_params)
    model.to(args.device)
    env = TSPEnv(**env_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    val_solver = TSPSolver()
    val_solver.from_txt(str(args.val_data), ref=True)

    print("=" * 72)
    print("LC Final Training - ELG-lite local policy")
    print("=" * 72)
    print(f"Run directory:             {run_dir}")
    print(f"Epochs:                    {args.epochs}")
    print(f"Batches per epoch:         {args.batches_per_epoch}")
    print(f"Batch size:                {args.batch_size}")
    print(f"Learning rate:             {args.learning_rate}")
    print(f"Validation interval:       {args.val_interval}")
    print(f"Local K:                   {args.local_k}")
    print(f"Local policy dim:          {args.local_policy_dim}")
    print(f"Local score weight:        {args.local_score_weight}")
    print(f"Global distance penalty:   {args.global_distance_penalty}")
    print(f"Distance K:                {args.distance_k}")
    print(f"DAR enabled during train:  False")
    print(f"Device:                    {args.device}")
    print("=" * 72)

    best_val_cost = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_length = 0.0
        running_best = 0.0
        running_loss = 0.0
        iterator = tqdm(range(args.batches_per_epoch), desc=f"Epoch {epoch:03d}/{args.epochs}", unit="batch", leave=False)
        for batch_idx in iterator:
            metrics = train_one_batch(model, env, optimizer, args.batch_size)
            running_length += metrics["avg_length"]
            running_best += metrics["best_length"]
            running_loss += metrics["loss"]
            iterator.set_postfix(length=f"{metrics['avg_length']:.4f}", best=f"{metrics['best_length']:.4f}", loss=f"{metrics['loss']:.4f}")
            if (batch_idx + 1) % max(1, args.batches_per_epoch // 4) == 0:
                print(
                    f"[train_lc] epoch={epoch} batch={batch_idx + 1}/{args.batches_per_epoch} "
                    f"avg_len={metrics['avg_length']:.4f} best_len={metrics['best_length']:.4f} loss={metrics['loss']:.4f}",
                    flush=True,
                )

        running_length /= args.batches_per_epoch
        running_best /= args.batches_per_epoch
        running_loss /= args.batches_per_epoch
        print(
            f"Epoch {epoch:03d}/{args.epochs} - Train Length: {running_length:.4f}, "
            f"Train Best: {running_best:.4f}, Loss: {running_loss:.4f}",
            end="",
            flush=True,
        )

        should_validate = epoch % args.val_interval == 0 or epoch == args.epochs
        if should_validate:
            val_cost = validate(model, env, val_solver, args.device)
            print(f" | Val Cost: {val_cost:.4f}", end="", flush=True)
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                torch.save(model.state_dict(), run_dir / "best_model.pth")
                torch.save(model.state_dict(), args.save_dir / "best_model.pth")
                torch.save(model.state_dict(), run_dir / f"model_epoch_{epoch}_cost_{val_cost:.4f}.pth")
                print(" | *** New best model saved ***", end="", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
