"""
Autoresearch-style MPC distillation script. Single-file, self-contained.

The goal is to train a neural network policy that imitates a finite-horizon
MPC teacher on a simple linear system, then evaluate the learned controller in
closed loop. Agents can tune architecture and optimization hyperparameters by
editing only this file.

Usage: uv run train_mpc.py
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60.0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Teacher / plant
DT = 0.1
STATE_DIM = 4
CONTROL_DIM = 2
HORIZON = 20
CONTROL_LIMIT = 2.0

# Search-relevant architecture knobs for agents
NUM_LAYERS = 3
HIDDEN_SIZE = 128
ACTIVATION = "tanh"  # relu | tanh | gelu

# Optimization
BATCH_SIZE = 512
MAX_STEPS = 100000
LR = 3e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
TRAIN_SPLIT = 0.9

# Dataset / evaluation
NUM_SAMPLES = 20000
EVAL_EPISODES = 128
ROLLOUT_STEPS = 80
STATE_STD = 2.0


# ---------------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------------

A = torch.tensor([
    [1.0, DT, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, DT],
    [0.0, 0.0, 0.0, 1.0],
], dtype=DTYPE)
B = torch.tensor([
    [0.5 * DT * DT, 0.0],
    [DT, 0.0],
    [0.0, 0.5 * DT * DT],
    [0.0, DT],
], dtype=DTYPE)
Q = torch.diag(torch.tensor([4.0, 0.5, 4.0, 0.5], dtype=DTYPE))
R = torch.diag(torch.tensor([0.2, 0.2], dtype=DTYPE))
QF = torch.diag(torch.tensor([8.0, 1.0, 8.0, 1.0], dtype=DTYPE))
REFERENCE_STATE = torch.zeros(STATE_DIM, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    state_dim: int = STATE_DIM
    control_dim: int = CONTROL_DIM
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    activation: str = ACTIVATION


class MLPPolicy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        activation = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }[config.activation]

        layers: list[nn.Module] = []
        in_dim = config.state_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_size))
            layers.append(activation())
            in_dim = config.hidden_size
        layers.append(nn.Linear(in_dim, config.control_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CONTROL_LIMIT * torch.tanh(self.net(x) / CONTROL_LIMIT)


# ---------------------------------------------------------------------------
# MPC teacher
# ---------------------------------------------------------------------------


def finite_horizon_lqr_gain(
    a: torch.Tensor,
    b: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    qf: torch.Tensor,
    horizon: int,
) -> list[torch.Tensor]:
    p = qf.clone()
    gains: list[torch.Tensor] = []
    for _ in range(horizon):
        gram = r + b.T @ p @ b
        k = torch.linalg.solve(gram, b.T @ p @ a)
        gains.append(k)
        p = q + a.T @ p @ (a - b @ k)
    gains.reverse()
    return gains


TEACHER_GAINS = finite_horizon_lqr_gain(A, B, Q, R, QF, HORIZON)


def mpc_teacher_action(x: torch.Tensor) -> torch.Tensor:
    k0 = TEACHER_GAINS[0].to(x.device)
    error = x - REFERENCE_STATE.to(x.device)
    u = -(error @ k0.T)
    return torch.clamp(u, -CONTROL_LIMIT, CONTROL_LIMIT)


# ---------------------------------------------------------------------------
# Data generation and metrics
# ---------------------------------------------------------------------------


def sample_states(num_samples: int, device: torch.device) -> torch.Tensor:
    return STATE_STD * torch.randn(num_samples, STATE_DIM, device=device, dtype=DTYPE)


@torch.no_grad()
def make_dataset(num_samples: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    states = sample_states(num_samples, device)
    actions = mpc_teacher_action(states)
    return states, actions


@torch.no_grad()
def rollout_policy(policy: nn.Module, initial_states: torch.Tensor, device: torch.device) -> dict[str, float]:
    a = A.to(device)
    b = B.to(device)
    q = Q.to(device)
    r = R.to(device)
    qf = QF.to(device)
    reference = REFERENCE_STATE.to(device)

    x = initial_states.clone()
    total_cost = torch.zeros(x.size(0), device=device, dtype=DTYPE)
    total_teacher_gap = torch.zeros_like(total_cost)

    for _ in range(ROLLOUT_STEPS):
        error = x - reference
        u = policy(x)
        teacher_u = mpc_teacher_action(x)
        total_teacher_gap += (u - teacher_u).square().mean(dim=1)
        stage_state = (error @ q * error).sum(dim=1)
        stage_control = (u @ r * u).sum(dim=1)
        total_cost += stage_state + stage_control
        x = x @ a.T + u @ b.T

    terminal_error = x - reference
    total_cost += (terminal_error @ qf * terminal_error).sum(dim=1)
    return {
        "eval_cost": total_cost.mean().item(),
        "teacher_mse": total_teacher_gap.mean().item() / ROLLOUT_STEPS,
        "state_rms": x.square().mean().sqrt().item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def get_activation_name() -> str:
    if ACTIVATION not in {"relu", "tanh", "gelu"}:
        raise ValueError(f"Unsupported ACTIVATION={ACTIVATION}")
    return ACTIVATION



def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = torch.device(DEVICE)
    config = PolicyConfig(activation=get_activation_name())
    model = MLPPolicy(config).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    states, actions = make_dataset(NUM_SAMPLES, device)
    num_train = int(TRAIN_SPLIT * NUM_SAMPLES)
    train_states, val_states = states[:num_train], states[num_train:]
    train_actions, val_actions = actions[:num_train], actions[num_train:]

    eval_initial_states = sample_states(EVAL_EPISODES, device)
    best_val_loss = math.inf
    smooth_loss = 0.0

    print("Policy config:")
    print(f"  device:          {device}")
    print(f"  model:           {asdict(config)}")
    print(f"  train_samples:   {num_train}")
    print(f"  val_samples:     {NUM_SAMPLES - num_train}")
    print(f"  horizon:         {HORIZON}")

    t_start = time.time()

    for step in range(MAX_STEPS):
        if time.time() - t_start >= TIME_BUDGET:
            break

        indices = torch.randint(0, num_train, (BATCH_SIZE,), device=device)
        xb = train_states[indices]
        yb = train_actions[indices]

        pred = model(xb)
        loss = F.mse_loss(pred, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        with torch.no_grad():
            val_pred = model(val_states)
            val_loss = F.mse_loss(val_pred, val_actions)
            best_val_loss = min(best_val_loss, val_loss.item())

        smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item()
        debiased = smooth_loss / (1 - 0.95 ** (step + 1))
        if step % 50 == 0:
            progress = min((time.time() - t_start) / TIME_BUDGET, 1.0)
            print(
                f"step {step:04d} ({100 * progress:5.1f}%) | "
                f"train_mse: {debiased:.6f} | val_mse: {val_loss.item():.6f}"
            )

    train_seconds = time.time() - t_start

    model.eval()
    with torch.no_grad():
        metrics = rollout_policy(model, eval_initial_states, device)

    num_params = sum(p.numel() for p in model.parameters())
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    print("---")
    print(f"eval_cost:        {metrics['eval_cost']:.6f}")
    print(f"teacher_mse:      {metrics['teacher_mse']:.6f}")
    print(f"state_rms:        {metrics['state_rms']:.6f}")
    print(f"best_val_mse:     {best_val_loss:.6f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params:       {num_params}")
    print(f"num_layers:       {NUM_LAYERS}")
    print(f"hidden_size:      {HIDDEN_SIZE}")


if __name__ == "__main__":
    main()
