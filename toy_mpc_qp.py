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

from toy_mpc_qp_utils import (
    INPUT_LIMIT,
    INPUT_RATE_LIMIT,
    POSITION_LIMIT,
    SCENARIO_SEED,
    VELOCITY_LIMIT,
    build_scenarios,
    NUM_SCENARIOS,
    resolve_device,
    simulate_closed_loop,
)


@dataclass
class MPCParams:
    prediction_horizon: int = 12
    control_horizon: int = 4
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    params = MPCParams()
    scenarios = build_scenarios(device=device)
    metrics = simulate_closed_loop(params, scenarios, device=device)

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
    print(f"scenario_config: num_scenarios={NUM_SCENARIOS} seed={SCENARIO_SEED}")
    print("---")
    print(f"objective: {metrics['objective']:.6f}")
    print(f"position_rmse: {metrics['position_rmse']:.6f}")
    print(f"velocity_rmse: {metrics['velocity_rmse']:.6f}")
    print(f"tracking_cost: {metrics['tracking_cost']:.6f}")
    print(f"input_cost: {metrics['input_cost']:.6f}")
    print(f"input_rate_cost: {metrics['input_rate_cost']:.6f}")
    print(f"constraint_violation: {metrics['constraint_violation']:.6f}")


if __name__ == "__main__":
    main()
