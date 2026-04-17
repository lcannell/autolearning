"""
Linear MPC benchmark with only 3 tunable hyperparameters.

This is a simpler variant of the main linear MPC benchmark. The dynamics,
constraints, scenario generation, and evaluation metric are kept explicit and
deterministic, but the controller design space is intentionally reduced.

Tunable hyperparameters:
    - horizon
    - control_weight
    - delta_u_weight

Everything else is fixed to keep the search space easier to understand.

Usage:
    uv run linear_mpc_problem_3param.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


# ---------------------------------------------------------------------------
# Fixed benchmark settings
# ---------------------------------------------------------------------------

DTYPE = torch.float64
DT = 0.2
STATE_DIM = 4
CONTROL_DIM = 1
ROLLOUT_STEPS = 40
SCENARIO_DIM = 4
SEARCH_NUM_SCENARIOS = 32
EVAL_NUM_SCENARIOS = 256
SEARCH_SCENARIO_SEED = 1234
EVAL_SCENARIO_SEED = 5678

CONTROL_LIMIT = 1.0
POSITION_LIMIT = 3.0
VELOCITY_LIMIT = 2.5

# Fixed outer-loop evaluation metric.
EVAL_POSITION_WEIGHT = 8.0
EVAL_VELOCITY_WEIGHT = 1.5
EVAL_CONTROL_WEIGHT = 0.15
EVAL_DELTA_U_WEIGHT = 0.35
EVAL_TERMINAL_MULTIPLIER = 3.0

# Formal domain for deterministic scenario generation.
POSITION_INIT_LIMIT = 2.5
VELOCITY_INIT_LIMIT = 1.5
REFERENCE_POSITION_LIMIT = 2.0
REFERENCE_VELOCITY_LIMIT = 0.25

# Fixed plant: double integrator tracking a moving reference.
A = torch.tensor([
    [1.0, DT, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, DT],
    [0.0, 0.0, 0.0, 1.0],
], dtype=DTYPE)
B = torch.tensor([
    [0.5 * DT * DT],
    [DT],
    [0.0],
    [0.0],
], dtype=DTYPE)


# ---------------------------------------------------------------------------
# Controller design hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class MPCParams:
    horizon: int = 8
    control_weight: float = 0.26
    delta_u_weight: float = 0.60


# Fixed internal design weights. These are no longer part of the search space.
DESIGN_POSITION_WEIGHT = 7.0
DESIGN_VELOCITY_WEIGHT = 2.2
DESIGN_TERMINAL_MULTIPLIER = 1.8


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


def build_scenarios(num_scenarios: int, seed: int) -> torch.Tensor:
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    engine = torch.quasirandom.SobolEngine(
        dimension=SCENARIO_DIM,
        scramble=True,
        seed=seed,
    )
    points = engine.draw(num_scenarios).to(dtype=DTYPE)

    lower = torch.tensor([
        -POSITION_INIT_LIMIT,
        -VELOCITY_INIT_LIMIT,
        -REFERENCE_POSITION_LIMIT,
        -REFERENCE_VELOCITY_LIMIT,
    ], dtype=DTYPE)
    upper = torch.tensor([
        POSITION_INIT_LIMIT,
        VELOCITY_INIT_LIMIT,
        REFERENCE_POSITION_LIMIT,
        REFERENCE_VELOCITY_LIMIT,
    ], dtype=DTYPE)
    return lower + (upper - lower) * points


SEARCH_SCENARIOS = build_scenarios(SEARCH_NUM_SCENARIOS, SEARCH_SCENARIO_SEED)
EVAL_SCENARIOS = build_scenarios(EVAL_NUM_SCENARIOS, EVAL_SCENARIO_SEED)


# ---------------------------------------------------------------------------
# MPC machinery
# ---------------------------------------------------------------------------


def validate_params(params: MPCParams) -> None:
    if params.horizon < 2:
        raise ValueError("horizon must be >= 2")
    if params.control_weight <= 0.0:
        raise ValueError("control_weight must be > 0")
    if params.delta_u_weight <= 0.0:
        raise ValueError("delta_u_weight must be > 0")


def make_augmented_system() -> tuple[torch.Tensor, torch.Tensor]:
    a_aug = torch.zeros(STATE_DIM + CONTROL_DIM, STATE_DIM + CONTROL_DIM, dtype=DTYPE)
    b_aug = torch.zeros(STATE_DIM + CONTROL_DIM, CONTROL_DIM, dtype=DTYPE)

    a_aug[:STATE_DIM, :STATE_DIM] = A
    a_aug[:STATE_DIM, STATE_DIM:] = B
    a_aug[STATE_DIM:, STATE_DIM:] = torch.eye(CONTROL_DIM, dtype=DTYPE)

    b_aug[:STATE_DIM, :] = B
    b_aug[STATE_DIM:, :] = torch.eye(CONTROL_DIM, dtype=DTYPE)
    return a_aug, b_aug


def make_cost_matrices(params: MPCParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.diag(torch.tensor([
        DESIGN_POSITION_WEIGHT,
        DESIGN_VELOCITY_WEIGHT,
        0.0,
        0.0,
        params.delta_u_weight,
    ], dtype=DTYPE))
    r = torch.diag(torch.tensor([params.control_weight], dtype=DTYPE))
    qf = DESIGN_TERMINAL_MULTIPLIER * q
    return q, r, qf


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


def make_tracking_error_state(x: torch.Tensor, previous_u: torch.Tensor) -> torch.Tensor:
    error = torch.empty(STATE_DIM + CONTROL_DIM, dtype=DTYPE)
    error[0] = x[0] - x[2]
    error[1] = x[1] - x[3]
    error[2] = 0.0
    error[3] = 0.0
    error[4] = previous_u
    return error


def clamp_state(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    raw_position_violation = torch.clamp(x[0].abs() - POSITION_LIMIT, min=0.0).item()
    raw_velocity_violation = torch.clamp(x[1].abs() - VELOCITY_LIMIT, min=0.0).item()
    x_next = x.clone()
    x_next[0] = torch.clamp(x_next[0], -POSITION_LIMIT, POSITION_LIMIT)
    x_next[1] = torch.clamp(x_next[1], -VELOCITY_LIMIT, VELOCITY_LIMIT)
    return x_next, raw_position_violation + raw_velocity_violation


def simulate_closed_loop(
    params: MPCParams,
    scenarios: torch.Tensor,
) -> dict[str, float]:
    validate_params(params)
    a_aug, b_aug = make_augmented_system()
    q, r, qf = make_cost_matrices(params)
    gains = finite_horizon_lqr_gain(a_aug, b_aug, q, r, qf, params.horizon)
    k0 = gains[0]

    total_tracking_cost = 0.0
    total_control_cost = 0.0
    total_slew_cost = 0.0
    total_constraint_violation = 0.0
    terminal_tracking_cost = 0.0

    for initial_state in scenarios:
        x = initial_state.clone()
        previous_u = torch.zeros(1, dtype=DTYPE)

        for _ in range(ROLLOUT_STEPS):
            error_state = make_tracking_error_state(x, previous_u)
            delta_u = -(k0 @ error_state.unsqueeze(1)).squeeze(1)
            u = torch.clamp(previous_u + delta_u, -CONTROL_LIMIT, CONTROL_LIMIT)
            applied_delta_u = u - previous_u

            tracking_error = torch.tensor([x[0] - x[2], x[1] - x[3]], dtype=DTYPE)
            total_tracking_cost += (
                EVAL_POSITION_WEIGHT * tracking_error[0].pow(2).item()
                + EVAL_VELOCITY_WEIGHT * tracking_error[1].pow(2).item()
            )
            total_control_cost += EVAL_CONTROL_WEIGHT * u.pow(2).sum().item()
            total_slew_cost += EVAL_DELTA_U_WEIGHT * applied_delta_u.pow(2).sum().item()

            x = A @ x + B @ u
            x, raw_violation = clamp_state(x)
            total_constraint_violation += raw_violation
            previous_u = u

        final_tracking_error = torch.tensor([x[0] - x[2], x[1] - x[3]], dtype=DTYPE)
        terminal_tracking_cost += EVAL_TERMINAL_MULTIPLIER * (
            EVAL_POSITION_WEIGHT * final_tracking_error[0].pow(2).item()
            + EVAL_VELOCITY_WEIGHT * final_tracking_error[1].pow(2).item()
        )

    normalizer = float(scenarios.size(0) * ROLLOUT_STEPS)
    objective = (
        total_tracking_cost
        + total_control_cost
        + total_slew_cost
        + terminal_tracking_cost
        + 500.0 * total_constraint_violation
    ) / normalizer

    return {
        "objective": objective,
        "tracking_cost": total_tracking_cost / normalizer,
        "control_cost": total_control_cost / normalizer,
        "slew_cost": total_slew_cost / normalizer,
        "terminal_cost": terminal_tracking_cost / normalizer,
        "constraint_violation": total_constraint_violation / normalizer,
    }


def evaluate_hyperparameters(params: MPCParams) -> float:
    return simulate_closed_loop(params, SEARCH_SCENARIOS)["objective"]


def main() -> None:
    params = MPCParams()
    search_metrics = simulate_closed_loop(params, SEARCH_SCENARIOS)
    eval_metrics = simulate_closed_loop(params, EVAL_SCENARIOS)

    print("Linear MPC benchmark (3 hyperparameters)")
    print(f"params: {asdict(params)}")
    print(
        "fixed_design: "
        f"position_weight={DESIGN_POSITION_WEIGHT}, "
        f"velocity_weight={DESIGN_VELOCITY_WEIGHT}, "
        f"terminal_multiplier={DESIGN_TERMINAL_MULTIPLIER}"
    )
    print(
        "scenario_config: "
        f"search={SEARCH_NUM_SCENARIOS} seed={SEARCH_SCENARIO_SEED}, "
        f"eval={EVAL_NUM_SCENARIOS} seed={EVAL_SCENARIO_SEED}"
    )
    print("---")
    print(f"objective: {search_metrics['objective']:.6f}")
    print(f"eval_objective: {eval_metrics['objective']:.6f}")
    print(f"search_tracking_cost: {search_metrics['tracking_cost']:.6f}")
    print(f"search_control_cost: {search_metrics['control_cost']:.6f}")
    print(f"search_slew_cost: {search_metrics['slew_cost']:.6f}")
    print(f"search_terminal_cost: {search_metrics['terminal_cost']:.6f}")
    print(f"search_constraint_violation: {search_metrics['constraint_violation']:.6f}")
    print(f"eval_tracking_cost: {eval_metrics['tracking_cost']:.6f}")
    print(f"eval_control_cost: {eval_metrics['control_cost']:.6f}")
    print(f"eval_slew_cost: {eval_metrics['slew_cost']:.6f}")
    print(f"eval_terminal_cost: {eval_metrics['terminal_cost']:.6f}")
    print(f"eval_constraint_violation: {eval_metrics['constraint_violation']:.6f}")


if __name__ == "__main__":
    main()
