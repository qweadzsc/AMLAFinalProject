"""Standalone LC/POMO trainer with optional Leader Reward.

This file keeps the original baseline files untouched and imports only the
baseline model/environment interfaces.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from ml4co_kit import TSPSolver
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
    parser = argparse.ArgumentParser(description="Train LC/POMO with optional Leader Reward.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--node-cnt", type=int, default=50)
    parser.add_argument("--pomo-size", type=int, default=50)
    parser.add_argument("--use-leader-reward", type=int, choices=[0, 1], default=1)
    parser.add_argument("--leader-reward-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--normalize-leader-advantage",
        type=int,
        choices=[0, 1],
        default=1,
        help="Divide all advantages by the LR multiplier, matching the paper's practical note.",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=REPO_ROOT / "Project" / "data" / "val" / "tsp50_uniform_val_128.txt",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "Project" / "experiments" / "lc_leader" / "checkpoints",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def configure_device(device: str) -> None:
    if "cuda" in device:
        torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")


def compute_loss(
    reward: torch.Tensor,
    log_prob: torch.Tensor,
    use_leader_reward: bool,
    leader_reward_multiplier: float,
    normalize_leader_advantage: bool,
) -> tuple:
    advantage = reward - reward.mean(dim=1, keepdim=True)
    pomo_loss = -(advantage.detach() * log_prob).mean()

    if not use_leader_reward:
        return pomo_loss, pomo_loss.detach(), torch.zeros((), device=reward.device), advantage.detach()

    leader_idx = reward.argmax(dim=1, keepdim=True)
    leader_mask = torch.zeros_like(reward, dtype=torch.bool)
    leader_mask.scatter_(1, leader_idx, True)

    leader_advantage = torch.where(
        leader_mask,
        advantage * leader_reward_multiplier,
        advantage,
    )
    if normalize_leader_advantage and leader_reward_multiplier > 0:
        leader_advantage = leader_advantage / leader_reward_multiplier

    loss = -(leader_advantage.detach() * log_prob).mean()
    leader_delta_loss = loss.detach() - pomo_loss.detach()
    return loss, pomo_loss.detach(), leader_delta_loss, leader_advantage.detach()


def rollout_batch(model, env, batch_size):
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

    log_prob = torch.stack(prob_list, dim=2).log().sum(dim=2)
    return reward, log_prob


def train_one_batch(model, env, optimizer, args):
    reward, log_prob = rollout_batch(model, env, args.batch_size)
    loss, pomo_loss, leader_delta_loss, leader_advantage = compute_loss(
        reward=reward,
        log_prob=log_prob,
        use_leader_reward=bool(args.use_leader_reward),
        leader_reward_multiplier=args.leader_reward_multiplier,
        normalize_leader_advantage=bool(args.normalize_leader_advantage),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "avg_length": -reward.mean().item(),
        "best_length": -reward.max(dim=1).values.mean().item(),
        "loss": loss.item(),
        "pomo_loss": pomo_loss.item(),
        "leader_delta_loss": leader_delta_loss.item(),
        "advantage_abs_mean": leader_advantage.abs().mean().item(),
    }


def validate(model, env, val_solver, device: str) -> float:
    model.eval()
    total_cost = 0.0

    with torch.no_grad():
        for i in range(len(val_solver.points)):
            coords = torch.from_numpy(val_solver.points[i : i + 1]).float().to(device)
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


def average_metrics(metrics):
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def main():
    args = parse_args()
    configure_device(args.device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = LCModel(**DEFAULT_MODEL_PARAMS)
    env = TSPEnv(task="TSP", node_cnt=args.node_cnt, pomo_size=args.pomo_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    val_solver = TSPSolver()
    val_solver.from_txt(str(args.val_data), ref=True)

    print("=" * 72)
    print("LC/POMO Training with Leader Reward")
    print("=" * 72)
    print(f"Run dir:                  {run_dir}")
    print(f"Epochs:                   {args.epochs}")
    print(f"Batches per epoch:        {args.batches_per_epoch}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Use Leader Reward:        {bool(args.use_leader_reward)}")
    print(f"LR multiplier:            {args.leader_reward_multiplier}")
    print(f"Normalize LR advantage:   {bool(args.normalize_leader_advantage)}")
    print(f"Device:                   {args.device}")
    print("=" * 72)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    best_val_cost = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_metrics = []
        pbar = tqdm(range(args.batches_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}", unit="batch", leave=False)
        for _ in pbar:
            batch_metrics = train_one_batch(model, env, optimizer, args)
            epoch_metrics.append(batch_metrics)
            pbar.set_postfix(
                {
                    "len": f"{batch_metrics['avg_length']:.4f}",
                    "loss": f"{batch_metrics['loss']:.4f}",
                    "pomo": f"{batch_metrics['pomo_loss']:.4f}",
                    "leader": f"{batch_metrics['leader_delta_loss']:.4f}",
                }
            )

        metrics = average_metrics(epoch_metrics)
        val_cost = validate(model, env, val_solver, args.device)
        metrics.update({"epoch": epoch, "val_cost": val_cost})
        history.append(metrics)

        print(
            f"Epoch {epoch:3d}/{args.epochs} - "
            f"Train Length: {metrics['avg_length']:.4f}, "
            f"Best Length: {metrics['best_length']:.4f}, "
            f"POMO Loss: {metrics['pomo_loss']:.4f}, "
            f"Leader Delta Loss: {metrics['leader_delta_loss']:.4f}, "
            f"Loss: {metrics['loss']:.4f}, "
            f"Val Cost: {val_cost:.4f}",
            end="",
        )

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            torch.save(model.state_dict(), run_dir / f"model_epoch_{epoch}_cost_{val_cost:.4f}.pth")
            print(" | *** New best model saved ***", end="")
        print()

        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    print("=" * 72)
    print(f"Training completed. Best validation cost: {best_val_cost:.4f}")
    print(f"Model saved to: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
