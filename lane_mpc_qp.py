"""
Lane-change MPC benchmark with obstacle vehicles.

This entrypoint mirrors the simpler toy MPC example:
- set the four calibration parameters in `MPCParams`
- run one closed-loop evaluation
- save a tracking plot and print the final objective

Usage:
    uv run lane_mpc_qp.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lane_mpc_qp_utils import (
    DT,
    LANE_1_W,
    LANE_2_W,
    MAX_LATERAL,
    MAX_SPEED,
    MAX_SPEED_RATE,
    MAX_STEERING,
    MAX_STEERING_RATE,
    MAX_HEADING,
    MIN_LATERAL,
    MIN_SPEED,
    MIN_STEERING,
    MIN_HEADING,
    build_scenarios,
    control_horizon_from_fraction,
    rollout_trajectory,
    simulate_closed_loop,
)


@dataclass
class MPCParams:
    prediction_horizon: int = 14
    control_horizon_fraction: float = 0.50
    q_delta_u_v: float = 0.10025884372280375
    q_delta_u_psi: float = 10.0


def save_tracking_plot(trajectory: dict[str, list], output_path: Path) -> None:
    states = np.asarray(trajectory["states"])
    inputs = np.asarray(trajectory["inputs"])
    steps = np.asarray(trajectory["steps"])
    time = steps * DT

    figure, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(states[:, 0], states[:, 1], label="SV", linewidth=2.0)
    axes[0].axhline(LANE_1_W, color="0.2", linestyle="--", linewidth=1.0)
    axes[0].axhline(LANE_2_W, color="0.2", linestyle="--", linewidth=1.0)
    for obstacle_index in range(len(trajectory["obstacles"][0])):
        obstacle_path = np.asarray([positions[obstacle_index] for positions in trajectory["obstacles"]])
        axes[0].plot(obstacle_path[:, 0], obstacle_path[:, 1], linewidth=1.5, label=f"OV {obstacle_index + 1}")
    axes[0].set_ylabel("lateral w_f [m]")
    axes[0].set_xlabel("longitudinal x_f [m]")
    axes[0].set_title("Lane-change tracking scenario")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, states[:, 1], label="w_f", linewidth=2.0)
    axes[1].plot(time, trajectory["reference_lateral"], label="w_ref", linewidth=2.0, linestyle="--")
    axes[1].set_ylabel("lateral [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time, inputs[:, 0], label="v [m/s]", linewidth=2.0)
    axes[2].plot(time, np.rad2deg(inputs[:, 1]), label="psi [deg]", linewidth=2.0)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("commands")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    params = MPCParams()
    scenarios = build_scenarios()
    metrics = simulate_closed_loop(params, scenarios)
    trajectory = rollout_trajectory(params, scenarios[0])
    plot_path = Path("lane_mpc_qp_tracking.png")
    save_tracking_plot(trajectory, plot_path)

    print("Lane-change MPC QP benchmark")
    print(f"params: {asdict(params)}")
    print(f"control_horizon: {control_horizon_from_fraction(params)}")
    print(
        "fixed_limits: "
        f"w=[{MIN_LATERAL}, {MAX_LATERAL}] m, "
        f"theta=[{np.rad2deg(MIN_HEADING):.1f}, {np.rad2deg(MAX_HEADING):.1f}] deg, "
        f"v=[{MIN_SPEED:.3f}, {MAX_SPEED:.3f}] m/s "
        f"([{MIN_SPEED * 3.6:.1f}, {MAX_SPEED * 3.6:.1f}] km/h), "
        f"psi=[{np.rad2deg(MIN_STEERING):.1f}, {np.rad2deg(MAX_STEERING):.1f}] deg, "
        f"v_dot=[{-MAX_SPEED_RATE}, {MAX_SPEED_RATE}] m/s^2, "
        f"psi_dot=[{-np.rad2deg(MAX_STEERING_RATE):.1f}, {np.rad2deg(MAX_STEERING_RATE):.1f}] deg/s"
    )
    print(f"scenario_config: num_scenarios={len(scenarios)}")
    print("---")
    print(f"objective: {metrics['objective']:.6f}")
    print(f"t_raise: {metrics['t_raise']:.6f}")
    print(f"settling_time: {metrics['settling_time']:.6f}")
    print(f"overshoot_cost: {metrics['overshoot_cost']:.6f}")
    print(f"collision_count: {metrics['collision_count']:.0f}")
    print(f"collision_cost: {metrics['collision_cost']:.6f}")
    print(f"tracking_plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
