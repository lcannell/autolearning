from __future__ import annotations

import math
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

REFERENCE_POSITION = 1.0
REFERENCE_VELOCITY = 0.0

RAISE_TIME_WEIGHT = 1.0
SETTLING_TIME_WEIGHT = 1.0
OVERSHOOT_EXP_SCALE = 30.0

QP_MAX_ITER = 4000
QP_ABS_TOL = 1e-6
QP_REL_TOL = 1e-6

ROLL_OUT_STEPS_DEFAULT = 40

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
    q_position: float
    q_velocity: float
    input_rate_weight: float


@dataclass
class Scenario:
    initial_state: torch.Tensor
    reference_position: float
    reference_velocity: float


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


def build_scenarios(device: str | torch.device | None = None) -> list[Scenario]:
    resolved_device = resolve_device(device)
    return [
        Scenario(
            initial_state=torch.zeros(STATE_DIM, dtype=DTYPE, device=resolved_device),
            reference_position=REFERENCE_POSITION,
            reference_velocity=REFERENCE_VELOCITY,
        )
    ]


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
    device: torch.device,
) -> torch.Tensor:
    lifting = torch.zeros(
        prediction_horizon * INPUT_DIM,
        prediction_horizon * INPUT_DIM,
        dtype=DTYPE,
        device=device,
    )
    for step in range(prediction_horizon):
        for increment_index in range(step + 1):
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

        self.state_transition, self.input_response = build_state_prediction_matrices(
            n_pred,
            self.device,
        )
        self.control_lifting = build_control_lifting_matrix(n_pred, self.device)
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
            n_pred * INPUT_DIM,
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
            (n_pred * INPUT_DIM,),
            -INPUT_RATE_LIMIT,
            dtype=DTYPE,
            device=self.device,
        )
        input_rate_max = torch.full(
            (n_pred * INPUT_DIM,),
            INPUT_RATE_LIMIT,
            dtype=DTYPE,
            device=self.device,
        )

        self.constraint_matrix = torch.cat(
            [
                self.predicted_output_map,
                self.control_lifting,
                torch.eye(n_pred * INPUT_DIM, dtype=DTYPE, device=self.device),
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

    def _reference_trajectory(
        self,
        reference_position: float,
        reference_velocity: float,
    ) -> torch.Tensor:
        reference_output = torch.tensor(
            [reference_position, reference_velocity],
            dtype=DTYPE,
            device=self.device,
        )
        return reference_output.repeat(self.params.prediction_horizon)

    def compute_control(
        self,
        current_state: torch.Tensor,
        previous_input: torch.Tensor,
        reference_position: float,
        reference_velocity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        free_output = (
            self.predicted_output_state_map @ current_state
            + self.predicted_output_previous_input_map @ previous_input
        )
        free_input = self.previous_input_map @ previous_input
        reference_trajectory = self._reference_trajectory(
            reference_position,
            reference_velocity,
        )

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


def simulate_closed_loop(
    params: MPCParameterLike,
    scenarios: list[Scenario],
    rollout_steps: int = ROLL_OUT_STEPS_DEFAULT,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    validate_params(params)
    total_raise_time = 0.0
    total_settling_time = 0.0
    total_overshoot_cost = 0.0

    for scenario in scenarios:
        trajectory = rollout_trajectory(
            params,
            scenario,
            rollout_steps=rollout_steps,
            device=device,
        )
        state_trajectory = trajectory["states"]
        reference_state = torch.tensor(
            [scenario.reference_position, scenario.reference_velocity],
            dtype=DTYPE,
        )
        initial_state = state_trajectory[0]
        target_delta = reference_state - initial_state
        target_distance = torch.linalg.vector_norm(target_delta).item()

        if target_distance <= 1e-9:
            raise_time = 0.0
            settling_radius = 0.0
            overshoot_cost = 0.0
        else:
            direction = target_delta / target_distance
            raise_time = float(rollout_steps)
            for step, state in enumerate(state_trajectory[1:], start=1):
                progress = torch.dot(state - initial_state, direction).item()
                if progress >= 0.9 * target_distance:
                    raise_time = float(step)
                    break

            settling_radius = 0.03 * target_distance
            max_normalized_overshoot = 0.0
            for state in state_trajectory:
                progress = torch.dot(state - initial_state, direction).item()
                normalized_progress = progress / target_distance
                max_normalized_overshoot = max(max_normalized_overshoot, normalized_progress - 1.0)
            overshoot_cost = math.exp(OVERSHOOT_EXP_SCALE * max_normalized_overshoot) - 1.0

        settling_time = float(rollout_steps)
        for step in range(len(state_trajectory)):
            stays_settled = True
            for future_state in state_trajectory[step:]:
                tracking_error = torch.linalg.vector_norm(future_state - reference_state).item()
                if tracking_error > settling_radius:
                    stays_settled = False
                    break
            if stays_settled:
                settling_time = float(step)
                break

        total_raise_time += raise_time
        total_settling_time += settling_time
        total_overshoot_cost += overshoot_cost

    num_scenarios = float(len(scenarios))
    average_raise_time = total_raise_time / num_scenarios
    average_settling_time = total_settling_time / num_scenarios
    average_overshoot_cost = total_overshoot_cost / num_scenarios
    objective = (
        RAISE_TIME_WEIGHT * average_raise_time
        + average_overshoot_cost
        + SETTLING_TIME_WEIGHT * average_settling_time
    )

    return {
        "objective": objective,
        "t_raise": average_raise_time,
        "settling_time": average_settling_time,
        "t_overshoot": average_overshoot_cost,
    }


def rollout_trajectory(
    params: MPCParameterLike,
    scenario: Scenario,
    rollout_steps: int = ROLL_OUT_STEPS_DEFAULT,
    device: str | torch.device | None = None,
) -> dict[str, list[int] | list[float] | list[torch.Tensor]]:
    validate_params(params)
    resolved_device = resolve_device(device)
    controller = ToyMPCController(params, resolved_device)
    a_matrix = A.to(device=resolved_device)
    b_matrix = B.to(device=resolved_device)

    controller.reset()
    state = scenario.initial_state.to(device=resolved_device).clone()
    previous_input = torch.zeros(INPUT_DIM, dtype=DTYPE, device=resolved_device)

    steps = list(range(rollout_steps + 1))
    positions = [float(state[0].item())]
    velocities = [float(state[1].item())]
    reference_positions = [scenario.reference_position]
    reference_velocities = [scenario.reference_velocity]
    states = [state.detach().cpu().clone()]

    for _ in range(rollout_steps):
        control_input, _ = controller.compute_control(
            state,
            previous_input,
            scenario.reference_position,
            scenario.reference_velocity,
        )
        state = a_matrix @ state + b_matrix @ control_input
        previous_input = control_input

        positions.append(float(state[0].item()))
        velocities.append(float(state[1].item()))
        reference_positions.append(scenario.reference_position)
        reference_velocities.append(scenario.reference_velocity)
        states.append(state.detach().cpu().clone())

    return {
        "steps": steps,
        "positions": positions,
        "velocities": velocities,
        "reference_positions": reference_positions,
        "reference_velocities": reference_velocities,
        "states": states,
    }
