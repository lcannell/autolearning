"""
Toy MPC example with a QP solved at each sampling time.

This file is the small entrypoint for the benchmark:
- choose the MPC hyperparameters manually in `MPCParams`
- choose the number of scenarios from the command line
- run one closed-loop evaluation and print the metrics

The implementation details live in `toy_mpc_qp_utils.py`.

Usage:
    uv run toy_mpc_qp.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from toy_mpc_qp_utils import (
    INPUT_LIMIT,
    INPUT_RATE_LIMIT,
    NUM_SCENARIOS_DEFAULT,
    POSITION_LIMIT,
    SCENARIO_SEED_DEFAULT,
    VELOCITY_LIMIT,
    MPCParams,
    build_scenarios,
    simulate_closed_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy MPC QP benchmark")
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=NUM_SCENARIOS_DEFAULT,
        help="number of scenarios",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCENARIO_SEED_DEFAULT,
        help="base random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = MPCParams()
    scenarios = build_scenarios(args.num_scenarios, args.seed)
    metrics = simulate_closed_loop(params, scenarios)

    print("Toy MPC QP benchmark")
    print(f"params: {asdict(params)}")
    print(
        "fixed_limits: "
        f"position={POSITION_LIMIT}, "
        f"velocity={VELOCITY_LIMIT}, "
        f"input={INPUT_LIMIT}, "
        f"delta_u={INPUT_RATE_LIMIT}"
    )
    print(f"scenario_config: num_scenarios={args.num_scenarios} seed={args.seed}")
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
