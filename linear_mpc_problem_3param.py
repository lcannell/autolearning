"""
Linear MPC benchmark with only 3 tunable hyperparameters.

This is a simpler variant of the main linear MPC benchmark. The dynamics,
constraints, scenario generation, and evaluation metric are kept explicit and
deterministic, but the controller design space is intentionally reduced.

Tunable hyperparameters:
    - prediction_horizon
    - control_horizon
    - input_rate_weight

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
PLANT_STATE_DIM = 4
CONTROL_DIM = 1
ROLLOUT_STEPS = 40
SCENARIO_DIM = PLANT_STATE_DIM
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
    prediction_horizon: int = 12
    control_horizon: int = 4
    input_rate_weight: float = 0.60


# Fixed MPC design weights and bounds. Only the three parameters above are
# tuned; everything else is kept fixed to mirror the formulation in the paper.
OUTPUT_TRACKING_WEIGHT_MATRIX = torch.diag(torch.tensor([7.0, 2.2], dtype=DTYPE))
INPUT_TRACKING_WEIGHT = 0.26
INPUT_REFERENCE = 0.0
CONTROL_INCREMENT_LIMIT = 0.35
QP_RHO = 1.0
QP_SIGMA = 1e-7
QP_MAX_ITER = 100
QP_ABS_TOL = 1e-7


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
    if params.prediction_horizon < 2:
        raise ValueError("prediction_horizon must be >= 2")
    if params.control_horizon < 1:
        raise ValueError("control_horizon must be >= 1")
    if params.control_horizon > params.prediction_horizon:
        raise ValueError("control_horizon must be <= prediction_horizon")
    if params.input_rate_weight <= 0.0:
        raise ValueError("input_rate_weight must be > 0")


def block_diag_repeat(block: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return torch.block_diag(*[block.clone() for _ in range(repeats)])


def build_state_prediction_matrices(
    prediction_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_transition = torch.zeros(
        prediction_horizon * PLANT_STATE_DIM,
        PLANT_STATE_DIM,
        dtype=DTYPE,
    )
    input_response = torch.zeros(
        prediction_horizon * PLANT_STATE_DIM,
        prediction_horizon * CONTROL_DIM,
        dtype=DTYPE,
    )

    state_power = torch.eye(PLANT_STATE_DIM, dtype=DTYPE)
    for step in range(prediction_horizon):
        state_power = state_power @ A
        row_slice = slice(step * PLANT_STATE_DIM, (step + 1) * PLANT_STATE_DIM)
        state_transition[row_slice, :] = state_power

        response_power = torch.eye(PLANT_STATE_DIM, dtype=DTYPE)
        for input_step in range(step, -1, -1):
            col_slice = slice(
                input_step * CONTROL_DIM,
                (input_step + 1) * CONTROL_DIM,
            )
            input_response[row_slice, col_slice] = response_power @ B
            response_power = response_power @ A

    return state_transition, input_response


def build_control_lifting_matrix(
    prediction_horizon: int,
    control_horizon: int,
) -> torch.Tensor:
    lifting = torch.zeros(
        prediction_horizon * CONTROL_DIM,
        control_horizon * CONTROL_DIM,
        dtype=DTYPE,
    )
    for step in range(prediction_horizon):
        last_active_increment = min(step, control_horizon - 1)
        for increment_index in range(last_active_increment + 1):
            row_slice = slice(step * CONTROL_DIM, (step + 1) * CONTROL_DIM)
            col_slice = slice(
                increment_index * CONTROL_DIM,
                (increment_index + 1) * CONTROL_DIM,
            )
            lifting[row_slice, col_slice] = torch.eye(CONTROL_DIM, dtype=DTYPE)
    return lifting


def compute_tracking_error(plant_state: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [plant_state[0] - plant_state[2], plant_state[1] - plant_state[3]],
        dtype=DTYPE,
    )


def compute_constraint_violation(plant_state: torch.Tensor) -> float:
    position_violation = torch.clamp(
        plant_state[0].abs() - POSITION_LIMIT,
        min=0.0,
    ).item()
    velocity_violation = torch.clamp(
        plant_state[1].abs() - VELOCITY_LIMIT,
        min=0.0,
    ).item()
    return position_violation + velocity_violation


def compute_input_violation(control_command: torch.Tensor) -> float:
    return torch.clamp(control_command.abs() - CONTROL_LIMIT, min=0.0).sum().item()


def compute_input_rate_violation(control_increment: torch.Tensor) -> float:
    return torch.clamp(
        control_increment.abs() - CONTROL_INCREMENT_LIMIT,
        min=0.0,
    ).sum().item()


class BoxConstrainedQPSolver:
    """
    Solve
        min 0.5 x^T P x + q^T x
        s.t. lower <= A x <= upper
    with a small-scale ADMM scheme tailored to this benchmark.
    """

    def __init__(self, p_matrix: torch.Tensor, a_matrix: torch.Tensor) -> None:
        self.p_matrix = p_matrix
        self.a_matrix = a_matrix
        regularizer = QP_SIGMA * torch.eye(p_matrix.size(0), dtype=DTYPE)
        self.linear_system_matrix = (
            p_matrix
            + QP_RHO * (a_matrix.T @ a_matrix)
            + regularizer
        )
        self.linear_system_cholesky = torch.linalg.cholesky(self.linear_system_matrix)
        self.x = torch.zeros(p_matrix.size(0), dtype=DTYPE)
        self.z = torch.zeros(a_matrix.size(0), dtype=DTYPE)
        self.y = torch.zeros(a_matrix.size(0), dtype=DTYPE)

    def reset(self) -> None:
        self.x.zero_()
        self.z.zero_()
        self.y.zero_()

    def solve(
        self,
        linear_term: torch.Tensor,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ) -> torch.Tensor:
        x = self.x.clone()
        z = self.z.clone()
        y = self.y.clone()

        for _ in range(QP_MAX_ITER):
            rhs = -linear_term + QP_RHO * self.a_matrix.T @ (z - y)
            x = torch.cholesky_solve(
                rhs.unsqueeze(1),
                self.linear_system_cholesky,
            ).squeeze(1)

            primal_argument = self.a_matrix @ x + y
            z_next = torch.minimum(torch.maximum(primal_argument, lower_bound), upper_bound)
            y = y + self.a_matrix @ x - z_next

            primal_residual = torch.max(torch.abs(self.a_matrix @ x - z_next)).item()
            dual_residual = torch.max(
                torch.abs(QP_RHO * self.a_matrix.T @ (z_next - z))
            ).item()
            z = z_next

            if primal_residual <= QP_ABS_TOL and dual_residual <= QP_ABS_TOL:
                break

        self.x = x
        self.z = z
        self.y = y
        return x


class RecedingHorizonController:
    def __init__(self, params: MPCParams) -> None:
        self.params = params
        prediction_horizon = params.prediction_horizon
        control_horizon = params.control_horizon

        self.state_transition, self.input_response = build_state_prediction_matrices(
            prediction_horizon,
        )
        self.control_lifting = build_control_lifting_matrix(
            prediction_horizon,
            control_horizon,
        )

        repeated_previous_control_map = torch.ones(
            prediction_horizon * CONTROL_DIM,
            CONTROL_DIM,
            dtype=DTYPE,
        )
        self.repeated_previous_control_map = repeated_previous_control_map
        self.input_map = self.control_lifting

        tracking_output_matrix = torch.tensor([
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=DTYPE)
        plant_output_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=DTYPE)

        tracking_output_stack = block_diag_repeat(
            tracking_output_matrix,
            prediction_horizon,
        )
        plant_output_stack = block_diag_repeat(
            plant_output_matrix,
            prediction_horizon,
        )

        self.predicted_tracking_error_map = tracking_output_stack @ self.input_response @ self.input_map
        self.predicted_tracking_error_state_map = tracking_output_stack @ self.state_transition
        self.predicted_tracking_error_previous_input_map = (
            tracking_output_stack @ self.input_response @ self.repeated_previous_control_map
        )

        self.predicted_output_map = plant_output_stack @ self.input_response @ self.input_map
        self.predicted_output_state_map = plant_output_stack @ self.state_transition
        self.predicted_output_previous_input_map = (
            plant_output_stack @ self.input_response @ self.repeated_previous_control_map
        )

        self.output_tracking_weights = block_diag_repeat(
            OUTPUT_TRACKING_WEIGHT_MATRIX,
            prediction_horizon,
        )
        self.input_tracking_weights = INPUT_TRACKING_WEIGHT * torch.eye(
            prediction_horizon * CONTROL_DIM,
            dtype=DTYPE,
        )
        input_rate_weights = params.input_rate_weight * torch.eye(
            control_horizon * CONTROL_DIM,
            dtype=DTYPE,
        )

        quadratic_term = (
            self.predicted_tracking_error_map.T
            @ self.output_tracking_weights
            @ self.predicted_tracking_error_map
            + self.input_map.T @ self.input_tracking_weights @ self.input_map
            + input_rate_weights
        )
        self.qp_hessian = 2.0 * quadratic_term

        output_min = torch.tensor(
            [-POSITION_LIMIT, -VELOCITY_LIMIT],
            dtype=DTYPE,
        ).repeat(prediction_horizon)
        output_max = torch.tensor(
            [POSITION_LIMIT, VELOCITY_LIMIT],
            dtype=DTYPE,
        ).repeat(prediction_horizon)
        input_min = torch.full(
            (prediction_horizon * CONTROL_DIM,),
            -CONTROL_LIMIT,
            dtype=DTYPE,
        )
        input_max = torch.full(
            (prediction_horizon * CONTROL_DIM,),
            CONTROL_LIMIT,
            dtype=DTYPE,
        )
        input_rate_min = torch.full(
            (control_horizon * CONTROL_DIM,),
            -CONTROL_INCREMENT_LIMIT,
            dtype=DTYPE,
        )
        input_rate_max = torch.full(
            (control_horizon * CONTROL_DIM,),
            CONTROL_INCREMENT_LIMIT,
            dtype=DTYPE,
        )

        self.constraint_matrix = torch.cat([
            self.predicted_output_map,
            self.input_map,
            torch.eye(control_horizon * CONTROL_DIM, dtype=DTYPE),
        ], dim=0)
        self.constraint_lower_template = torch.cat([
            output_min,
            input_min,
            input_rate_min,
        ])
        self.constraint_upper_template = torch.cat([
            output_max,
            input_max,
            input_rate_max,
        ])
        self.qp_solver = BoxConstrainedQPSolver(
            self.qp_hessian,
            self.constraint_matrix,
        )

    def compute_control(
        self,
        plant_state: torch.Tensor,
        previous_control: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        free_tracking_error = (
            self.predicted_tracking_error_state_map @ plant_state
            + self.predicted_tracking_error_previous_input_map @ previous_control
        )
        free_input = self.repeated_previous_control_map @ previous_control
        free_output = (
            self.predicted_output_state_map @ plant_state
            + self.predicted_output_previous_input_map @ previous_control
        )

        linear_term = 2.0 * (
            self.predicted_tracking_error_map.T
            @ self.output_tracking_weights
            @ free_tracking_error
            + self.input_map.T
            @ self.input_tracking_weights
            @ (free_input - INPUT_REFERENCE)
        )

        lower_bound = self.constraint_lower_template.clone()
        upper_bound = self.constraint_upper_template.clone()

        output_constraint_size = self.params.prediction_horizon * 2
        input_constraint_size = self.params.prediction_horizon * CONTROL_DIM

        lower_bound[:output_constraint_size] -= free_output
        upper_bound[:output_constraint_size] -= free_output
        lower_bound[
            output_constraint_size:output_constraint_size + input_constraint_size
        ] -= free_input
        upper_bound[
            output_constraint_size:output_constraint_size + input_constraint_size
        ] -= free_input

        optimal_increment_sequence = self.qp_solver.solve(
            linear_term,
            lower_bound,
            upper_bound,
        )
        control_increment = optimal_increment_sequence[:CONTROL_DIM]
        control_command = previous_control + control_increment
        applied_control_increment = control_command - previous_control
        return control_command, applied_control_increment


def simulate_closed_loop(
    params: MPCParams,
    scenarios: torch.Tensor,
) -> dict[str, float]:
    validate_params(params)
    controller = RecedingHorizonController(params)

    total_tracking_cost = 0.0
    total_control_cost = 0.0
    total_input_rate_cost = 0.0
    total_constraint_violation = 0.0
    terminal_tracking_cost = 0.0

    for initial_state in scenarios:
        plant_state = initial_state.clone()
        previous_control = torch.zeros(1, dtype=DTYPE)

        for _ in range(ROLLOUT_STEPS):
            control_command, applied_control_increment = controller.compute_control(
                plant_state,
                previous_control,
            )
            tracking_error = compute_tracking_error(plant_state)
            total_tracking_cost += (
                EVAL_POSITION_WEIGHT * tracking_error[0].pow(2).item()
                + EVAL_VELOCITY_WEIGHT * tracking_error[1].pow(2).item()
            )
            total_control_cost += EVAL_CONTROL_WEIGHT * control_command.pow(2).sum().item()
            total_input_rate_cost += (
                EVAL_DELTA_U_WEIGHT * applied_control_increment.pow(2).sum().item()
            )

            plant_state = A @ plant_state + B @ control_command
            total_constraint_violation += (
                compute_constraint_violation(plant_state)
                + compute_input_violation(control_command)
                + compute_input_rate_violation(applied_control_increment)
            )
            previous_control = control_command

        final_tracking_error = compute_tracking_error(plant_state)
        terminal_tracking_cost += EVAL_TERMINAL_MULTIPLIER * (
            EVAL_POSITION_WEIGHT * final_tracking_error[0].pow(2).item()
            + EVAL_VELOCITY_WEIGHT * final_tracking_error[1].pow(2).item()
        )

    normalizer = float(scenarios.size(0) * ROLLOUT_STEPS)
    objective = (
        total_tracking_cost
        + total_control_cost
        + total_input_rate_cost
        + terminal_tracking_cost
        + 500.0 * total_constraint_violation
    ) / normalizer

    return {
        "objective": objective,
        "tracking_cost": total_tracking_cost / normalizer,
        "control_cost": total_control_cost / normalizer,
        "slew_cost": total_input_rate_cost / normalizer,
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
        f"output_tracking_weights={OUTPUT_TRACKING_WEIGHT_MATRIX.diag().tolist()}, "
        f"input_tracking_weight={INPUT_TRACKING_WEIGHT}, "
        f"delta_u_limit={CONTROL_INCREMENT_LIMIT}"
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
