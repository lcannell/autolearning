from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import osqp
import torch
from scipy import sparse


DTYPE = torch.float64
DT = 0.2
STATE_DIM = 2
INPUT_DIM = 1
OUTPUT_DIM = 2

POSITION_LIMIT = 4.0
VELOCITY_LIMIT = 3.0
INPUT_LIMIT = 1.25
INPUT_RATE_LIMIT = 0.50

POSITION_INIT_LIMIT = 2.0
VELOCITY_INIT_LIMIT = 1.0
REFERENCE_POSITION_LIMIT = 2.5

POSITION_WEIGHT = 10.0
VELOCITY_WEIGHT = 1.0
INPUT_WEIGHT = 0.10
INPUT_RATE_WEIGHT = 0.25
CONSTRAINT_PENALTY = 200.0

QP_MAX_ITER = 4000
QP_ABS_TOL = 1e-6
QP_REL_TOL = 1e-6

ROLL_OUT_STEPS_DEFAULT = 40
NUM_SCENARIOS = 24
SCENARIO_SEED = 1234

CPU_DEVICE = torch.device("cpu")

A = torch.tensor([
    [1.0, DT],
    [0.0, 1.0],
], dtype=DTYPE, device=CPU_DEVICE)
B = torch.tensor([
    [0.5 * DT * DT],
    [DT],
], dtype=DTYPE, device=CPU_DEVICE)
C = torch.eye(OUTPUT_DIM, dtype=DTYPE, device=CPU_DEVICE)


class MPCParameterLike(Protocol):
    prediction_horizon: int
    control_horizon: int
    q_position: float
    q_velocity: float
    input_rate_weight: float


@dataclass
class Scenario:
    initial_state: torch.Tensor
    reference_position: float


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{resolved}', but CUDA is not available")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device '{resolved}', but only {torch.cuda.device_count()} CUDA device(s) are available"
            )
    return resolved


def validate_params(params: MPCParameterLike) -> None:
    if params.prediction_horizon < 2:
        raise ValueError("prediction_horizon must be >= 2")
    if params.control_horizon < 1:
        raise ValueError("control_horizon must be >= 1")
    if params.control_horizon > params.prediction_horizon:
        raise ValueError("control_horizon must be <= prediction_horizon")
    if params.q_position <= 0.0:
        raise ValueError("q_position must be > 0")
    if params.q_velocity <= 0.0:
        raise ValueError("q_velocity must be > 0")
    if params.input_rate_weight <= 0.0:
        raise ValueError("input_rate_weight must be > 0")


def block_diag_repeat(block: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return torch.block_diag(*[block.clone() for _ in range(repeats)])


def build_scenarios(
    seed: int = SCENARIO_SEED,
    num_scenarios: int = NUM_SCENARIOS,
    device: str | torch.device | None = None,
) -> list[Scenario]:
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    resolved_device = resolve_device(device)
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    points = engine.draw(num_scenarios).to(dtype=DTYPE, device=resolved_device)

    lower = torch.tensor(
        [-POSITION_INIT_LIMIT, -VELOCITY_INIT_LIMIT, -REFERENCE_POSITION_LIMIT],
        dtype=DTYPE,
        device=resolved_device,
    )
    upper = torch.tensor(
        [POSITION_INIT_LIMIT, VELOCITY_INIT_LIMIT, REFERENCE_POSITION_LIMIT],
        dtype=DTYPE,
        device=resolved_device,
    )
    values = lower + (upper - lower) * points

    scenarios: list[Scenario] = []
    for row in values:
        scenarios.append(
            Scenario(
                initial_state=row[:2].clone(),
                reference_position=float(row[2].item()),
            )
        )
    return scenarios


def build_state_prediction_matrices(
    prediction_horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_transition = torch.zeros(
        prediction_horizon * STATE_DIM,
        STATE_DIM,
        dtype=DTYPE,
        device=device,
    )
    input_response = torch.zeros(
        prediction_horizon * STATE_DIM,
        prediction_horizon * INPUT_DIM,
        dtype=DTYPE,
        device=device,
    )

    a_matrix = A.to(device=device)
    b_matrix = B.to(device=device)

    for step in range(prediction_horizon):
        state_power = torch.matrix_power(a_matrix, step + 1)
        row_slice = slice(step * STATE_DIM, (step + 1) * STATE_DIM)
        state_transition[row_slice, :] = state_power

        for input_step in range(step + 1):
            col_slice = slice(input_step * INPUT_DIM, (input_step + 1) * INPUT_DIM)
            input_response[row_slice, col_slice] = (
                torch.matrix_power(a_matrix, step - input_step) @ b_matrix
            )

    return state_transition, input_response


def build_control_lifting_matrix(
    prediction_horizon: int,
    control_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    lifting = torch.zeros(
        prediction_horizon * INPUT_DIM,
        control_horizon * INPUT_DIM,
        dtype=DTYPE,
        device=device,
    )
    for step in range(prediction_horizon):
        last_increment_index = min(step, control_horizon - 1)
        for increment_index in range(last_increment_index + 1):
            row_slice = slice(step * INPUT_DIM, (step + 1) * INPUT_DIM)
            col_slice = slice(increment_index * INPUT_DIM, (increment_index + 1) * INPUT_DIM)
            lifting[row_slice, col_slice] = torch.eye(INPUT_DIM, dtype=DTYPE, device=device)
    return lifting


class BoxConstrainedQPSolver:
    """Solve `0.5 x^T P x + q^T x` with box-constrained linear inequalities via OSQP."""

    def __init__(self, p_matrix: torch.Tensor, a_matrix: torch.Tensor, device: torch.device) -> None:
        self.device = device
        p_numpy = p_matrix.detach().cpu().numpy()
        p_sparse = sparse.csc_matrix(0.5 * (p_numpy + p_numpy.T))
        a_sparse = sparse.csc_matrix(a_matrix.detach().cpu().numpy())

        self.problem = osqp.OSQP()
        self.problem.setup(
            P=p_sparse,
            q=np.zeros(p_matrix.size(0), dtype=np.float64),
            A=a_sparse,
            l=np.full(a_matrix.size(0), -np.inf, dtype=np.float64),
            u=np.full(a_matrix.size(0), np.inf, dtype=np.float64),
            verbose=False,
            warm_start=True,
            polish=False,
            max_iter=QP_MAX_ITER,
            eps_abs=QP_ABS_TOL,
            eps_rel=QP_REL_TOL,
        )
        self.last_solution = np.zeros(p_matrix.size(0), dtype=np.float64)

    def reset_warm_start(self) -> None:
        self.last_solution.fill(0.0)

    def solve(
        self,
        linear_term: torch.Tensor,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ) -> torch.Tensor:
        q = linear_term.detach().cpu().numpy().astype(np.float64, copy=False)
        l = lower_bound.detach().cpu().numpy().astype(np.float64, copy=False)
        u = upper_bound.detach().cpu().numpy().astype(np.float64, copy=False)

        self.problem.update(q=q, l=l, u=u)
        self.problem.warm_start(x=self.last_solution)
        result = self.problem.solve()

        if result.x is None or result.info.status not in {"solved", "solved inaccurate"}:
            raise RuntimeError(f"OSQP failed to solve the MPC QP: {result.info.status}")

        self.last_solution = result.x.copy()
        return torch.tensor(result.x, dtype=DTYPE, device=self.device)


class ToyMPCController:
    def __init__(self, params: MPCParameterLike, device: str | torch.device | None = None) -> None:
        self.params = params
        self.device = resolve_device(device)

        n_pred = params.prediction_horizon
        n_ctrl = params.control_horizon

        self.state_transition, self.input_response = build_state_prediction_matrices(
            n_pred,
            self.device,
        )
        self.control_lifting = build_control_lifting_matrix(n_pred, n_ctrl, self.device)
        self.previous_input_map = torch.ones(
            n_pred * INPUT_DIM,
            INPUT_DIM,
            dtype=DTYPE,
            device=self.device,
        )

        output_stack = block_diag_repeat(C.to(device=self.device), n_pred)
        self.predicted_output_map = output_stack @ self.input_response @ self.control_lifting
        self.predicted_output_state_map = output_stack @ self.state_transition
        self.predicted_output_previous_input_map = (
            output_stack @ self.input_response @ self.previous_input_map
        )

        self.output_weights = block_diag_repeat(
            torch.diag(
                torch.tensor(
                    [params.q_position, params.q_velocity],
                    dtype=DTYPE,
                    device=self.device,
                )
            ),
            n_pred,
        )
        self.input_weights = 0.15 * torch.eye(n_pred * INPUT_DIM, dtype=DTYPE, device=self.device)
        self.input_rate_weights = params.input_rate_weight * torch.eye(
            n_ctrl * INPUT_DIM,
            dtype=DTYPE,
            device=self.device,
        )

        quadratic_term = (
            self.predicted_output_map.T @ self.output_weights @ self.predicted_output_map
            + self.control_lifting.T @ self.input_weights @ self.control_lifting
            + self.input_rate_weights
        )
        self.qp_hessian = 2.0 * quadratic_term

        output_min = torch.tensor(
            [-POSITION_LIMIT, -VELOCITY_LIMIT],
            dtype=DTYPE,
            device=self.device,
        ).repeat(n_pred)
        output_max = torch.tensor(
            [POSITION_LIMIT, VELOCITY_LIMIT],
            dtype=DTYPE,
            device=self.device,
        ).repeat(n_pred)
        input_min = torch.full((n_pred * INPUT_DIM,), -INPUT_LIMIT, dtype=DTYPE, device=self.device)
        input_max = torch.full((n_pred * INPUT_DIM,), INPUT_LIMIT, dtype=DTYPE, device=self.device)
        input_rate_min = torch.full(
            (n_ctrl * INPUT_DIM,),
            -INPUT_RATE_LIMIT,
            dtype=DTYPE,
            device=self.device,
        )
        input_rate_max = torch.full(
            (n_ctrl * INPUT_DIM,),
            INPUT_RATE_LIMIT,
            dtype=DTYPE,
            device=self.device,
        )

        self.constraint_matrix = torch.cat(
            [
                self.predicted_output_map,
                self.control_lifting,
                torch.eye(n_ctrl * INPUT_DIM, dtype=DTYPE, device=self.device),
            ],
            dim=0,
        )
        self.constraint_lower_template = torch.cat([output_min, input_min, input_rate_min])
        self.constraint_upper_template = torch.cat([output_max, input_max, input_rate_max])

        self.qp_solver = BoxConstrainedQPSolver(
            self.qp_hessian,
            self.constraint_matrix,
            self.device,
        )

    def reset(self) -> None:
        self.qp_solver.reset_warm_start()

    def _reference_trajectory(self, reference_position: float) -> torch.Tensor:
        reference_output = torch.tensor([reference_position, 0.0], dtype=DTYPE, device=self.device)
        return reference_output.repeat(self.params.prediction_horizon)

    def compute_control(
        self,
        current_state: torch.Tensor,
        previous_input: torch.Tensor,
        reference_position: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        free_output = (
            self.predicted_output_state_map @ current_state
            + self.predicted_output_previous_input_map @ previous_input
        )
        free_input = self.previous_input_map @ previous_input
        reference_trajectory = self._reference_trajectory(reference_position)

        linear_term = 2.0 * (
            self.predicted_output_map.T
            @ self.output_weights
            @ (free_output - reference_trajectory)
            + self.control_lifting.T @ self.input_weights @ free_input
        )

        lower_bound = self.constraint_lower_template.clone()
        upper_bound = self.constraint_upper_template.clone()

        output_constraint_size = self.params.prediction_horizon * OUTPUT_DIM
        input_constraint_size = self.params.prediction_horizon * INPUT_DIM

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
        control_increment = optimal_increment_sequence[:INPUT_DIM]
        control_input = previous_input + control_increment
        return control_input, control_increment


def compute_state_violation(state: torch.Tensor) -> float:
    position_violation = torch.clamp(state[0].abs() - POSITION_LIMIT, min=0.0).item()
    velocity_violation = torch.clamp(state[1].abs() - VELOCITY_LIMIT, min=0.0).item()
    return position_violation + velocity_violation


def compute_input_violation(control_input: torch.Tensor) -> float:
    return torch.clamp(control_input.abs() - INPUT_LIMIT, min=0.0).sum().item()


def compute_input_rate_violation(control_increment: torch.Tensor) -> float:
    return torch.clamp(control_increment.abs() - INPUT_RATE_LIMIT, min=0.0).sum().item()


def simulate_closed_loop(
    params: MPCParameterLike,
    scenarios: list[Scenario],
    rollout_steps: int = ROLL_OUT_STEPS_DEFAULT,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    validate_params(params)
    resolved_device = resolve_device(device)
    controller = ToyMPCController(params, resolved_device)
    a_matrix = A.to(device=resolved_device)
    b_matrix = B.to(device=resolved_device)

    total_tracking_cost = 0.0
    total_input_cost = 0.0
    total_input_rate_cost = 0.0
    total_constraint_violation = 0.0
    squared_position_error_sum = 0.0
    squared_velocity_error_sum = 0.0

    for scenario in scenarios:
        controller.reset()
        state = scenario.initial_state.to(device=resolved_device).clone()
        previous_input = torch.zeros(INPUT_DIM, dtype=DTYPE, device=resolved_device)

        for _ in range(rollout_steps):
            control_input, control_increment = controller.compute_control(
                state,
                previous_input,
                scenario.reference_position,
            )

            tracking_error = torch.tensor(
                [state[0] - scenario.reference_position, state[1]],
                dtype=DTYPE,
            )
            squared_position_error_sum += tracking_error[0].pow(2).item()
            squared_velocity_error_sum += tracking_error[1].pow(2).item()

            total_tracking_cost += (
                POSITION_WEIGHT * tracking_error[0].pow(2).item()
                + VELOCITY_WEIGHT * tracking_error[1].pow(2).item()
            )
            total_input_cost += INPUT_WEIGHT * control_input.pow(2).sum().item()
            total_input_rate_cost += (
                INPUT_RATE_WEIGHT * control_increment.pow(2).sum().item()
            )

            state = a_matrix @ state + b_matrix @ control_input
            total_constraint_violation += (
                compute_state_violation(state)
                + compute_input_violation(control_input)
                + compute_input_rate_violation(control_increment)
            )
            previous_input = control_input

    num_samples = float(len(scenarios) * rollout_steps)
    objective = (
        total_tracking_cost
        + total_input_cost
        + total_input_rate_cost
        + CONSTRAINT_PENALTY * total_constraint_violation
    ) / num_samples

    return {
        "objective": objective,
        "tracking_cost": total_tracking_cost / num_samples,
        "input_cost": total_input_cost / num_samples,
        "input_rate_cost": total_input_rate_cost / num_samples,
        "constraint_violation": total_constraint_violation / num_samples,
        "position_rmse": (squared_position_error_sum / num_samples) ** 0.5,
        "velocity_rmse": (squared_velocity_error_sum / num_samples) ** 0.5,
    }
