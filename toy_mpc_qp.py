"""
Toy MPC example with a QP solved at each sampling time.

This file is the small entrypoint for the benchmark:
- choose the MPC hyperparameters manually in `MPCParams`
- run one closed-loop evaluation and print the metrics

The implementation details live in `toy_mpc_qp_utils.py`.

Usage:
    uv run toy_mpc_qp.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_mpc_qp_utils import (
    INPUT_LIMIT,
    INPUT_RATE_LIMIT,
    POSITION_LIMIT,
    VELOCITY_LIMIT,
    build_scenarios,
    resolve_device,
    rollout_trajectory,
    simulate_closed_loop,
)


@dataclass
class MPCParams:
    prediction_horizon: int = 12
    q_position: float = 8.0
    q_velocity: float = 1.5
    input_rate_weight: float = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy MPC QP benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device to use, for example 'cuda:0' or 'cpu' (default: cuda:0 if available, else cpu)",
    )
    return parser.parse_args()


def save_tracking_plot(
    steps: list[int],
    positions: list[float],
    reference_positions: list[float],
    velocities: list[float],
    reference_velocities: list[float],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(steps, positions, label="position", linewidth=2.0)
    axes[0].plot(steps, reference_positions, label="reference", linewidth=2.0, linestyle="--")
    axes[0].set_ylabel("position")
    axes[0].set_title("Closed-loop tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, velocities, label="velocity", linewidth=2.0)
    axes[1].plot(steps, reference_velocities, label="reference", linewidth=2.0, linestyle="--")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("velocity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    params = MPCParams()
    scenarios = build_scenarios(device=device)
    metrics = simulate_closed_loop(params, scenarios, device=device)
    trajectory = rollout_trajectory(params, scenarios[0], device=device)
    plot_path = Path("toy_mpc_qp_tracking.png")
    save_tracking_plot(
        trajectory["steps"],
        trajectory["positions"],
        trajectory["reference_positions"],
        trajectory["velocities"],
        trajectory["reference_velocities"],
        plot_path,
    )

    print("Toy MPC QP benchmark")
    print(f"device: {device}")
    print(f"params: {asdict(params)}")
    print(
        "fixed_limits: "
        f"position={POSITION_LIMIT}, "
        f"velocity={VELOCITY_LIMIT}, "
        f"input={INPUT_LIMIT}, "
        f"delta_u={INPUT_RATE_LIMIT}"
    )
    print(f"scenario_config: num_scenarios={len(scenarios)}")
    print("---")
    print(f"objective: {metrics['objective']:.6f}")
    print(f"t_raise: {metrics['t_raise']:.6f}")
    print(f"settling_time: {metrics['settling_time']:.6f}")
    print(f"t_overshoot: {metrics['t_overshoot']:.6f}")
    print(f"tracking_plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
