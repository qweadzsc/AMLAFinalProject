"""Train LC/POMO with an ELG-lite local policy."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ml4co_kit import TSPSolver
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
LC_BASELINE_DIR = REPO_ROOT / "Project" / "baselines" / "lc_baseline"
LOCAL_MODEL_DIR = THIS_DIR / "model"
sys.path.insert(0, str(LC_BASELINE_DIR))
sys.path.insert(0, str(LOCAL_MODEL_DIR))

from model import LCModel, TSPEnv  # noqa: E402
from dar_wrapper import compute_dar_bias, compute_lc_logits, logits_to_probs  # noqa: E402
from local_policy import LocalPolicyScorer, build_visited_mask_from_ninf  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Train LC/POMO with ELG-lite local policy.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--node-cnt", type=int, default=50)
    parser.add_argument("--pomo-size", type=int, default=50)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--local-policy-dim", type=int, default=128)
    parser.add_argument("--local-score-weight", type=float, default=1.0)
    parser.add_argument("--global-distance-penalty", type=float, default=0.0)
    parser.add_argument("--distance-k", type=int, default=None)
    parser.add_argument("--joint-train", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pretrain-global-epochs", type=int, default=0)
    parser.add_argument("--dar-log-nearest", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--val-data",
        type=Path,
        default=REPO_ROOT / "Project" / "data" / "val" / "tsp50_uniform_val_128.txt",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "Project" / "experiments" / "lc_dar_elg" / "checkpoints",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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


def cuda_info(device: str) -> dict:
    info = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "requested_device": device,
    }
    if torch.cuda.is_available() and "cuda" in device:
        idx = int(device.split(":")[1]) if ":" in device else 0
        info["torch_current_device"] = torch.cuda.current_device()
        info["torch_device_name"] = torch.cuda.get_device_name(idx)
    return info


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def compute_pomo_loss(reward: torch.Tensor, log_prob: torch.Tensor):
    advantage = reward - reward.mean(dim=1, keepdim=True)
    loss = -(advantage.detach() * log_prob).mean()
    return loss, advantage.detach()




def finite_abs_mean(tensor: torch.Tensor) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return 0.0
    return finite.abs().mean().item()

def prepare_first_step(model: LCModel, state, coordinates: torch.Tensor):
    batch_size = state.BATCH_IDX.size(0)
    pomo_size = state.BATCH_IDX.size(1)
    selected = torch.arange(pomo_size, device=coordinates.device)[None, :].expand(batch_size, pomo_size)
    encoded_first_node = model.encoded_nodes.gather(
        dim=1,
        index=selected[:, :, None].expand(batch_size, pomo_size, model.encoded_nodes.size(2)),
    )
    model.decoder.set_q1(encoded_first_node)
    prob = torch.ones((batch_size, pomo_size), device=coordinates.device)
    return selected, prob


def compute_ensemble_logits(model, local_policy, env, state, args, epoch_idx: int):
    global_logits = compute_lc_logits(model, state)
    visited_mask = build_visited_mask_from_ninf(state.ninf_mask)
    local_score = local_policy(env.coordinates, state.current_node, visited_mask)

    distance_penalty = torch.zeros_like(global_logits)
    if args.global_distance_penalty != 0:
        distance_penalty = compute_dar_bias(
            coordinates=env.coordinates,
            current_node=state.current_node,
            ninf_mask=state.ninf_mask,
            dar_k=args.distance_k or args.local_k,
            dar_log_nearest=bool(args.dar_log_nearest),
        )
        distance_penalty = args.global_distance_penalty * distance_penalty

    effective_local_weight = args.local_score_weight if epoch_idx > args.pretrain_global_epochs else 0.0
    ensemble_logits = global_logits + distance_penalty + effective_local_weight * local_score
    stats = {
        "global_abs_mean": finite_abs_mean(global_logits),
        "local_abs_mean": finite_abs_mean(local_score),
        "distance_abs_mean": finite_abs_mean(distance_penalty),
        "ensemble_abs_mean": finite_abs_mean(ensemble_logits),
        "effective_local_weight": effective_local_weight,
    }
    return ensemble_logits, stats


def rollout_batch(model, local_policy, env, args, epoch_idx: int, greedy: bool):
    env.load_problems(args.batch_size)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    state, reward, done = env.pre_step()
    prob_list = []
    step_stats = []

    while not done:
        if state.current_node is None:
            selected, prob = prepare_first_step(model, state, reset_state.coordinates)
        else:
            ensemble_logits, stats = compute_ensemble_logits(model, local_policy, env, state, args, epoch_idx)
            probs = logits_to_probs(ensemble_logits)
            if greedy:
                selected = probs.argmax(dim=2)
                prob = None
            else:
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(args.batch_size * args.pomo_size, -1).multinomial(1)
                        selected = selected.squeeze(1).reshape(args.batch_size, args.pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(args.batch_size, args.pomo_size)
                    if (prob != 0).all():
                        break
            step_stats.append(stats)
        state, reward, done = env.step(selected)
        if prob is not None:
            prob_list.append(prob)

    log_prob = torch.stack(prob_list, dim=2).log().sum(dim=2)
    return reward, log_prob, step_stats


def average_step_stats(step_stats):
    if not step_stats:
        return {
            "global_abs_mean": 0.0,
            "local_abs_mean": 0.0,
            "distance_abs_mean": 0.0,
            "ensemble_abs_mean": 0.0,
            "effective_local_weight": 0.0,
        }
    keys = step_stats[0].keys()
    return {key: sum(item[key] for item in step_stats) / len(step_stats) for key in keys}


def train_one_batch(model, local_policy, env, optimizer, args, epoch_idx: int):
    reward, log_prob, step_stats = rollout_batch(model, local_policy, env, args, epoch_idx, greedy=False)
    loss, advantage = compute_pomo_loss(reward, log_prob)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    stats = average_step_stats(step_stats)
    stats.update(
        {
            "avg_length": -reward.mean().item(),
            "best_length": -reward.max(dim=1).values.mean().item(),
            "loss": loss.item(),
            "advantage_abs_mean": advantage.abs().mean().item(),
        }
    )
    return stats


def validate(model, local_policy, env, val_solver, args, epoch_idx: int) -> float:
    model.eval()
    local_policy.eval()
    total_cost = 0.0
    print(f"[train_lc_elg] starting validation for epoch={epoch_idx}", flush=True)

    with torch.no_grad():
        iterator = tqdm(
            range(len(val_solver.points)),
            desc=f"Validate {epoch_idx:03d}",
            unit="instance",
            disable=args.disable_progress,
            leave=False,
        )
        for i in iterator:
            coords = torch.from_numpy(val_solver.points[i : i + 1]).float().to(args.device)
            problems = torch.cdist(coords, coords, p=2)
            env.load_problems_manual(problems, coords)

            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                if state.current_node is None:
                    selected, _ = prepare_first_step(model, state, reset_state.coordinates)
                else:
                    ensemble_logits, _ = compute_ensemble_logits(model, local_policy, env, state, args, epoch_idx)
                    selected = logits_to_probs(ensemble_logits).argmax(dim=2)
                state, reward, done = env.step(selected)

            total_cost += -reward.max().item()

    model.train()
    local_policy.train()
    return total_cost / len(val_solver.points)


def average_metrics(metrics):
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def make_optimizer(model, local_policy, args):
    params = list(local_policy.parameters())
    if bool(args.joint_train):
        params += list(model.parameters())
        set_trainable(model, True)
    else:
        set_trainable(model, False)
    return torch.optim.Adam(params, lr=args.learning_rate)


def main():
    args = parse_args()
    set_seed(args.seed)
    configure_device(args.device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = LCModel(**DEFAULT_MODEL_PARAMS)
    local_policy = LocalPolicyScorer(
        hidden_dim=args.local_policy_dim,
        local_k=args.local_k,
        max_positional_rank=max(128, args.node_cnt),
        non_neighbor_value=0.0,
    )
    model.to(args.device)
    local_policy.to(args.device)
    env = TSPEnv(task="TSP", node_cnt=args.node_cnt, pomo_size=args.pomo_size)
    optimizer = make_optimizer(model, local_policy, args)

    val_solver = TSPSolver()
    val_solver.from_txt(str(args.val_data), ref=True)

    print("=" * 80)
    print("LC/POMO Training with ELG-lite Local Policy")
    print("=" * 80)
    print(f"Run dir:                  {run_dir}")
    print(f"Epochs:                   {args.epochs}")
    print(f"Batches per epoch:        {args.batches_per_epoch}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Seed:                     {args.seed}")
    print(f"Validation interval:      {args.val_interval}")
    print(f"Local K:                  {args.local_k}")
    print(f"Local policy dim:         {args.local_policy_dim}")
    print(f"Local score weight:       {args.local_score_weight}")
    print(f"Global distance penalty:  {args.global_distance_penalty}")
    print(f"Distance K:               {args.distance_k or args.local_k}")
    print(f"Joint train:              {bool(args.joint_train)}")
    print(f"Pretrain global epochs:   {args.pretrain_global_epochs}")
    print(f"Device:                   {args.device}")
    cuda_runtime_info = cuda_info(args.device)
    print(f"CUDA_VISIBLE_DEVICES:     {cuda_runtime_info['cuda_visible_devices']}")
    print(f"Torch CUDA devices:       {cuda_runtime_info['torch_cuda_device_count']}")
    if cuda_runtime_info.get("torch_device_name"):
        print(f"Torch device name:        {cuda_runtime_info['torch_device_name']}")
    print("=" * 80)

    config = vars(args).copy()
    config["cuda_info"] = cuda_runtime_info
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best_val_cost = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"[train_lc_elg] starting epoch={epoch}/{args.epochs}", flush=True)
        model.train()
        local_policy.train()
        epoch_metrics = []
        pbar = tqdm(
            range(args.batches_per_epoch),
            desc=f"Epoch {epoch:3d}/{args.epochs}",
            unit="batch",
            leave=False,
            disable=args.disable_progress,
        )
        for batch_idx in pbar:
            batch_metrics = train_one_batch(model, local_policy, env, optimizer, args, epoch)
            epoch_metrics.append(batch_metrics)
            pbar.set_postfix(
                {
                    "len": f"{batch_metrics['avg_length']:.4f}",
                    "loss": f"{batch_metrics['loss']:.4f}",
                    "global": f"{batch_metrics['global_abs_mean']:.3f}",
                    "local": f"{batch_metrics['local_abs_mean']:.3f}",
                    "dist": f"{batch_metrics['distance_abs_mean']:.3f}",
                }
            )
            if (batch_idx + 1) % max(1, args.batches_per_epoch // 4) == 0:
                print(
                    f"[train_lc_elg] epoch={epoch} batch={batch_idx + 1}/{args.batches_per_epoch} "
                    f"loss={batch_metrics['loss']:.4f} avg_len={batch_metrics['avg_length']:.4f} "
                    f"global_abs={batch_metrics['global_abs_mean']:.4f} "
                    f"local_abs={batch_metrics['local_abs_mean']:.4f}",
                    flush=True,
                )

        metrics = average_metrics(epoch_metrics)
        should_validate = epoch % args.val_interval == 0 or epoch == args.epochs
        val_cost = validate(model, local_policy, env, val_solver, args, epoch) if should_validate else None
        metrics.update({"epoch": epoch, "val_cost": val_cost})
        history.append(metrics)

        message = (
            f"Epoch {epoch:3d}/{args.epochs} - "
            f"Train Length: {metrics['avg_length']:.4f}, "
            f"Best Length: {metrics['best_length']:.4f}, "
            f"Loss: {metrics['loss']:.4f}, "
            f"Global Score Abs: {metrics['global_abs_mean']:.4f}, "
            f"Local Score Abs: {metrics['local_abs_mean']:.4f}, "
            f"Distance Abs: {metrics['distance_abs_mean']:.4f}, "
            f"Effective Local Weight: {metrics['effective_local_weight']:.2f}"
        )
        if val_cost is not None:
            message += f", Val Cost: {val_cost:.4f}"
        print(message, end="")

        if val_cost is not None and val_cost < best_val_cost:
            best_val_cost = val_cost
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "local_policy_state_dict": local_policy.state_dict(),
                "model_params": DEFAULT_MODEL_PARAMS,
                "local_policy_params": {
                    "hidden_dim": args.local_policy_dim,
                    "local_k": args.local_k,
                    "max_positional_rank": max(128, args.node_cnt),
                    "non_neighbor_value": 0.0,
                },
                "args": vars(args),
            }
            torch.save(checkpoint, run_dir / "best_model.pth")
            torch.save(checkpoint, run_dir / f"model_epoch_{epoch}_cost_{val_cost:.4f}.pth")
            print(" | *** New best model saved ***", end="")
        print()

        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    print("=" * 80)
    print(f"Training completed. Best validation cost: {best_val_cost:.4f}")
    print(f"Artifacts saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
