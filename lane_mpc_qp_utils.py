from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import osqp
from scipy import sparse


DT = 0.085
STATE_DIM = 3
INPUT_DIM = 2
OUTPUT_DIM = STATE_DIM

LANE_1_W = 0.0
LANE_2_W = 3.0

REFERENCE_SPEED = 50.0 / 3.6
REFERENCE_STEERING = 0.0

MIN_LATERAL = -0.6
MAX_LATERAL = 3.6
MIN_HEADING = math.radians(-90.0)
MAX_HEADING = math.radians(90.0)
MIN_SPEED = 1.0 / 3.6
MAX_SPEED = 90.0 / 3.6
MIN_STEERING = math.radians(-45.0)
MAX_STEERING = math.radians(45.0)
MAX_SPEED_RATE = 4.0
MAX_STEERING_RATE = math.radians(60.0)

WHEELBASE = 2.8
ROLL_OUT_STEPS_DEFAULT = 90

Q_Y = np.diag([0.0, 10.0, 1.0])
Q_U = np.diag([1.0, 1.0])

RAISE_TIME_WEIGHT = 1.0
SETTLING_TIME_WEIGHT = 1.0
LANE_ERROR_WEIGHT = 4.0
HEADING_ERROR_WEIGHT = 1.0
CONTROL_EFFORT_WEIGHT = 0.02
OVERSHOOT_WEIGHT = 10.0
COLLISION_PENALTY = 1.0e6

QP_MAX_ITER = 4000
QP_ABS_TOL = 1e-6
QP_REL_TOL = 1e-6
MIN_LOG_DELTA_U_WEIGHT = -5.0
MAX_LOG_DELTA_U_WEIGHT = 3.0


class MPCParameterLike(Protocol):
    prediction_horizon: int
    control_horizon_fraction: float
    q_delta_u_v: float
    q_delta_u_psi: float


@dataclass(frozen=True)
class ObstacleVehicle:
    initial_x: float
    lane_w: float
    speed: float

    def state_at_step(self, step: int) -> tuple[float, float]:
        return self.initial_x + self.speed * DT * step, self.lane_w


@dataclass(frozen=True)
class Scenario:
    initial_state: np.ndarray
    initial_input: np.ndarray
    reference_lane_w: float
    obstacles: tuple[ObstacleVehicle, ...]


def validate_params(params: MPCParameterLike) -> None:
    if not 10 <= params.prediction_horizon <= 30:
        raise ValueError("prediction_horizon must be in [10, 30]")
    if not 0.1 <= params.control_horizon_fraction <= 1.0:
        raise ValueError("control_horizon_fraction must be in [0.1, 1.0]")
    min_delta_u_weight = math.exp(MIN_LOG_DELTA_U_WEIGHT)
    max_delta_u_weight = math.exp(MAX_LOG_DELTA_U_WEIGHT)
    if not min_delta_u_weight <= params.q_delta_u_v <= max_delta_u_weight:
        raise ValueError("log(q_delta_u_v) must be in [-5, 3]")
    if not min_delta_u_weight <= params.q_delta_u_psi <= max_delta_u_weight:
        raise ValueError("log(q_delta_u_psi) must be in [-5, 3]")


def control_horizon_from_fraction(params: MPCParameterLike) -> int:
    return int(np.clip(round(params.control_horizon_fraction * params.prediction_horizon), 1, params.prediction_horizon))


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            initial_state=np.array([0.0, LANE_1_W, 0.0], dtype=np.float64),
            initial_input=np.array([REFERENCE_SPEED, REFERENCE_STEERING], dtype=np.float64),
            reference_lane_w=LANE_2_W,
            obstacles=(
                ObstacleVehicle(initial_x=6.0, lane_w=LANE_1_W, speed=38.0 / 3.6),
                ObstacleVehicle(initial_x=27.0, lane_w=LANE_2_W, speed=40.0 / 3.6),
            ),
        )
    ]


def vehicle_step(state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
    x_f, w_f, theta = state
    speed, steering = control_input
    travel_angle = theta + steering
    next_state = np.array(
        [
            x_f + DT * speed * math.cos(travel_angle),
            w_f + DT * speed * math.sin(travel_angle),
            theta + DT * speed * math.sin(steering) / WHEELBASE,
        ],
        dtype=np.float64,
    )
    next_state[2] = float(np.clip(next_state[2], MIN_HEADING, MAX_HEADING))
    return next_state


def linearize_dynamics(state: np.ndarray, control_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, _, theta = state
    speed, steering = control_input
    travel_angle = theta + steering

    a_matrix = np.eye(STATE_DIM, dtype=np.float64)
    a_matrix[0, 2] = -DT * speed * math.sin(travel_angle)
    a_matrix[1, 2] = DT * speed * math.cos(travel_angle)

    b_matrix = np.zeros((STATE_DIM, INPUT_DIM), dtype=np.float64)
    b_matrix[0, 0] = DT * math.cos(travel_angle)
    b_matrix[0, 1] = -DT * speed * math.sin(travel_angle)
    b_matrix[1, 0] = DT * math.sin(travel_angle)
    b_matrix[1, 1] = DT * speed * math.cos(travel_angle)
    b_matrix[2, 0] = DT * math.sin(steering) / WHEELBASE
    b_matrix[2, 1] = DT * speed * math.cos(steering) / WHEELBASE
    return a_matrix, b_matrix


def build_control_horizon_lifting(prediction_horizon: int, control_horizon: int) -> np.ndarray:
    lifting = np.zeros((prediction_horizon * INPUT_DIM, control_horizon * INPUT_DIM), dtype=np.float64)
    for step in range(prediction_horizon):
        held_step = min(step, control_horizon - 1)
        row = slice(step * INPUT_DIM, (step + 1) * INPUT_DIM)
        col = slice(held_step * INPUT_DIM, (held_step + 1) * INPUT_DIM)
        lifting[row, col] = np.eye(INPUT_DIM)
    return lifting


def build_difference_operator(horizon: int) -> np.ndarray:
    difference = np.zeros((horizon * INPUT_DIM, horizon * INPUT_DIM), dtype=np.float64)
    for step in range(horizon):
        row = slice(step * INPUT_DIM, (step + 1) * INPUT_DIM)
        col = slice(step * INPUT_DIM, (step + 1) * INPUT_DIM)
        difference[row, col] = np.eye(INPUT_DIM)
        if step > 0:
            previous_col = slice((step - 1) * INPUT_DIM, step * INPUT_DIM)
            difference[row, previous_col] = -np.eye(INPUT_DIM)
    return difference


def build_nominal_trajectory(
    initial_state: np.ndarray,
    initial_input: np.ndarray,
    prediction_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((prediction_horizon + 1, STATE_DIM), dtype=np.float64)
    inputs = np.zeros((prediction_horizon, INPUT_DIM), dtype=np.float64)
    states[0] = initial_state
    nominal_input = initial_input.copy()
    reference_input = np.array([REFERENCE_SPEED, REFERENCE_STEERING], dtype=np.float64)
    max_input_step = np.array([MAX_SPEED_RATE * DT, MAX_STEERING_RATE * DT], dtype=np.float64)
    for step in range(prediction_horizon):
        input_step = np.clip(reference_input - nominal_input, -max_input_step, max_input_step)
        nominal_input = nominal_input + input_step
        nominal_input[0] = float(np.clip(nominal_input[0], MIN_SPEED, MAX_SPEED))
        nominal_input[1] = float(np.clip(nominal_input[1], MIN_STEERING, MAX_STEERING))
        inputs[step] = nominal_input
        states[step + 1] = vehicle_step(states[step], inputs[step])
    return states, inputs


class LaneMPCController:
    def __init__(self, params: MPCParameterLike) -> None:
        validate_params(params)
        self.params = params
        self.n_pred = params.prediction_horizon
        self.n_ctrl = control_horizon_from_fraction(params)
        self.last_solution = np.zeros(self.n_ctrl * INPUT_DIM, dtype=np.float64)

    def reset(self) -> None:
        self.last_solution.fill(0.0)

    def compute_control(
        self,
        current_state: np.ndarray,
        previous_input: np.ndarray,
        reference_lane_w: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        nominal_states, nominal_inputs = build_nominal_trajectory(current_state, previous_input, self.n_pred)
        lifting = build_control_horizon_lifting(self.n_pred, self.n_ctrl)
        difference = build_difference_operator(self.n_pred)

        a_matrices = []
        b_matrices = []
        for step in range(self.n_pred):
            a_matrix, b_matrix = linearize_dynamics(nominal_states[step], nominal_inputs[step])
            a_matrices.append(a_matrix)
            b_matrices.append(b_matrix)

        initial_deviation = current_state - nominal_states[0]
        state_deviation_free = np.zeros(self.n_pred * STATE_DIM, dtype=np.float64)
        full_input_response = np.zeros((self.n_pred * STATE_DIM, self.n_pred * INPUT_DIM), dtype=np.float64)
        for output_step in range(self.n_pred):
            state_row = slice(output_step * STATE_DIM, (output_step + 1) * STATE_DIM)

            transition = np.eye(STATE_DIM, dtype=np.float64)
            for step in range(output_step, -1, -1):
                transition = transition @ a_matrices[step]
            state_deviation_free[state_row] = transition @ initial_deviation

            for input_step in range(output_step + 1):
                response = b_matrices[input_step]
                for step in range(input_step + 1, output_step + 1):
                    response = a_matrices[step] @ response
                input_col = slice(input_step * INPUT_DIM, (input_step + 1) * INPUT_DIM)
                full_input_response[state_row, input_col] = response

        nominal_state_stack = nominal_states[1:].reshape(self.n_pred * STATE_DIM)
        nominal_input_stack = nominal_inputs.reshape(self.n_pred * INPUT_DIM)

        predicted_state_map = full_input_response @ lifting
        predicted_input_map = lifting
        state_stack = nominal_state_stack + state_deviation_free
        input_stack = nominal_input_stack
        delta_input_map = difference @ predicted_input_map
        nominal_delta_input = difference @ input_stack
        nominal_delta_input[:INPUT_DIM] -= previous_input

        reference_state = np.tile(np.array([current_state[0] + REFERENCE_SPEED * DT, reference_lane_w, 0.0]), self.n_pred)
        for step in range(self.n_pred):
            reference_state[step * OUTPUT_DIM] = current_state[0] + REFERENCE_SPEED * DT * (step + 1)

        reference_input = np.tile(np.array([REFERENCE_SPEED, REFERENCE_STEERING]), self.n_pred)
        output_weights = sparse.block_diag([Q_Y] * self.n_pred, format="csc").toarray()
        input_weights = sparse.block_diag([Q_U] * self.n_pred, format="csc").toarray()
        delta_u_weights = sparse.block_diag(
            [np.diag([self.params.q_delta_u_v, self.params.q_delta_u_psi])] * self.n_pred,
            format="csc",
        ).toarray()

        p_matrix = (
            predicted_state_map.T @ output_weights @ predicted_state_map
            + predicted_input_map.T @ input_weights @ predicted_input_map
            + delta_input_map.T @ delta_u_weights @ delta_input_map
        )
        q_vector = (
            predicted_state_map.T @ output_weights @ (state_stack - reference_state)
            + predicted_input_map.T @ input_weights @ (input_stack - reference_input)
            + delta_input_map.T @ delta_u_weights @ nominal_delta_input
        )

        constraint_matrix = []
        lower_bounds = []
        upper_bounds = []

        state_min = np.tile(np.array([-np.inf, MIN_LATERAL, MIN_HEADING]), self.n_pred)
        state_max = np.tile(np.array([np.inf, MAX_LATERAL, MAX_HEADING]), self.n_pred)
        constraint_matrix.append(predicted_state_map)
        lower_bounds.append(state_min - state_stack)
        upper_bounds.append(state_max - state_stack)

        input_min = np.tile(np.array([MIN_SPEED, MIN_STEERING]), self.n_pred)
        input_max = np.tile(np.array([MAX_SPEED, MAX_STEERING]), self.n_pred)
        constraint_matrix.append(predicted_input_map)
        lower_bounds.append(input_min - input_stack)
        upper_bounds.append(input_max - input_stack)

        rate_limit = np.tile(np.array([MAX_SPEED_RATE * DT, MAX_STEERING_RATE * DT]), self.n_pred)
        constraint_matrix.append(delta_input_map)
        lower_bounds.append(-rate_limit - nominal_delta_input)
        upper_bounds.append(rate_limit - nominal_delta_input)

        a_constraint = np.vstack(constraint_matrix)
        lower_bound = np.concatenate(lower_bounds)
        upper_bound = np.concatenate(upper_bounds)

        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix(2.0 * p_matrix),
            q=2.0 * q_vector,
            A=sparse.csc_matrix(a_constraint),
            l=lower_bound,
            u=upper_bound,
            verbose=False,
            warm_start=True,
            polish=False,
            max_iter=QP_MAX_ITER,
            eps_abs=QP_ABS_TOL,
            eps_rel=QP_REL_TOL,
        )
        problem.warm_start(x=self.last_solution)
        result = problem.solve()
        if result.x is None or result.info.status not in {"solved", "solved inaccurate"}:
            raise RuntimeError(f"OSQP failed to solve the lane MPC QP: {result.info.status}")

        self.last_solution = result.x.copy()
        control_input = nominal_inputs[0] + result.x[:INPUT_DIM]
        control_input[0] = float(np.clip(control_input[0], MIN_SPEED, MAX_SPEED))
        control_input[1] = float(np.clip(control_input[1], MIN_STEERING, MAX_STEERING))
        control_increment = control_input - previous_input
        return control_input, control_increment


def collision_margin(
    sv_state: np.ndarray,
    obstacle_position: tuple[float, float],
    longitudinal_radius: float = 4.5,
    lateral_radius: float = 1.0,
) -> float:
    dx = abs(sv_state[0] - obstacle_position[0]) / longitudinal_radius
    dw = abs(sv_state[1] - obstacle_position[1]) / lateral_radius
    return max(dx, dw)


def rollout_trajectory(
    params: MPCParameterLike,
    scenario: Scenario,
    rollout_steps: int = ROLL_OUT_STEPS_DEFAULT,
) -> dict[str, list]:
    controller = LaneMPCController(params)
    controller.reset()

    state = scenario.initial_state.astype(np.float64).copy()
    previous_input = scenario.initial_input.astype(np.float64).copy()

    trajectory = {
        "steps": list(range(rollout_steps + 1)),
        "states": [state.copy()],
        "inputs": [previous_input.copy()],
        "increments": [np.zeros(INPUT_DIM, dtype=np.float64)],
        "reference_lateral": [scenario.reference_lane_w],
        "obstacles": [[obstacle.state_at_step(0) for obstacle in scenario.obstacles]],
        "collisions": [False],
    }

    for step in range(rollout_steps):
        control_input, control_increment = controller.compute_control(
            state,
            previous_input,
            scenario.reference_lane_w,
        )
        state = vehicle_step(state, control_input)
        previous_input = control_input

        obstacle_positions = [obstacle.state_at_step(step + 1) for obstacle in scenario.obstacles]
        has_collision = any(collision_margin(state, obstacle) <= 1.0 for obstacle in obstacle_positions)

        trajectory["states"].append(state.copy())
        trajectory["inputs"].append(control_input.copy())
        trajectory["increments"].append(control_increment.copy())
        trajectory["reference_lateral"].append(scenario.reference_lane_w)
        trajectory["obstacles"].append(obstacle_positions)
        trajectory["collisions"].append(has_collision)

    return trajectory


def simulate_closed_loop(
    params: MPCParameterLike,
    scenarios: list[Scenario],
    rollout_steps: int = ROLL_OUT_STEPS_DEFAULT,
) -> dict[str, float]:
    validate_params(params)
    total_raise_time = 0.0
    total_settling_time = 0.0
    total_overshoot_cost = 0.0
    total_collision_count = 0.0

    for scenario in scenarios:
        trajectory = rollout_trajectory(params, scenario, rollout_steps=rollout_steps)
        states = np.asarray(trajectory["states"])
        lateral_error = states[:, 1] - scenario.reference_lane_w

        raise_time = float(rollout_steps)
        lateral_distance = abs(scenario.reference_lane_w - scenario.initial_state[1])
        if lateral_distance <= 1e-9:
            raise_time = 0.0
        else:
            for step, error in enumerate(abs(lateral_error)):
                if error <= 0.1 * lateral_distance:
                    raise_time = float(step)
                    break

        settling_radius = max(0.03 * max(lateral_distance, 1.0), 0.05)
        settling_time = float(rollout_steps)
        for step in range(len(lateral_error)):
            if np.all(np.abs(lateral_error[step:]) <= settling_radius):
                settling_time = float(step)
                break

        travel_direction = math.copysign(1.0, scenario.reference_lane_w - scenario.initial_state[1])
        overshoot = np.maximum(travel_direction * lateral_error, 0.0)
        overshoot_cost = float(OVERSHOOT_WEIGHT * np.max(overshoot) ** 2)
        collision_count = float(sum(bool(value) for value in trajectory["collisions"]))

        total_raise_time += raise_time
        total_settling_time += settling_time
        total_overshoot_cost += overshoot_cost
        total_collision_count += collision_count

    num_scenarios = float(len(scenarios))
    average_raise_time = total_raise_time / num_scenarios
    average_settling_time = total_settling_time / num_scenarios
    average_overshoot_cost = total_overshoot_cost / num_scenarios
    average_collision_count = total_collision_count / num_scenarios
    collision_cost = COLLISION_PENALTY * average_collision_count
    objective = (
        RAISE_TIME_WEIGHT * average_raise_time
        + SETTLING_TIME_WEIGHT * average_settling_time
        + average_overshoot_cost
        + collision_cost
    )

    return {
        "objective": objective,
        "t_raise": average_raise_time,
        "settling_time": average_settling_time,
        "overshoot_cost": average_overshoot_cost,
        "collision_count": average_collision_count,
        "collision_cost": collision_cost,
    }
