# autoresearch-linear-mpc

This is an experiment to have the LLM do autonomous research on a direct linear
MPC benchmark.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: The repo is small. Read these files for context:
   - `README.md` — repository context.
   - `linear_mpc_problem.py` — the single benchmark file you modify.
   - `program_linear_mpc.md` — these instructions.
4. **Initialize results**: Ensure `results_linear_mpc.tsv` exists with just the header row if it is missing.
5. **Confirm and go**: confirm the setup looks correct before starting the search loop.

Once setup is confirmed, start iterating.

## Experimentation

Each experiment is a direct hyperparameter search over a deterministic linear
MPC benchmark. Launch a run with:

```bash
uv run linear_mpc_problem.py > run.log 2>&1
```

The script evaluates a constrained linear tracking controller in closed loop on
a fixed set of scenarios and prints a scalar objective to minimize.

**What you CAN do:**
- Modify `linear_mpc_problem.py` only.
- Tune the exposed MPC hyperparameters such as:
  - `horizon`
  - `position_weight`
  - `velocity_weight`
  - `control_weight`
  - `delta_u_weight`
  - `terminal_multiplier`
- Refine the benchmark internals only if it preserves the intended scope:
  deterministic linear MPC, constrained closed-loop evaluation, and a single
  scalar objective for comparison.

**What you CANNOT do:**
- Add dependencies.
- Spread the benchmark across multiple files unless the user explicitly asks.
- Turn it back into a neural network imitation problem.
- Change the benchmark in ways that destroy comparability across runs.

## Goal

**The goal is simple: get the lowest `objective`.** Lower is better.

Treat this as a controller-design search problem, not a training problem. Since
the benchmark is deterministic, small improvements are meaningful.

Secondary metrics:
- `tracking_cost`
- `control_cost`
- `slew_cost`
- `terminal_cost`
- `constraint_violation`

Prefer solutions that improve `objective` without exploding control effort or
constraint violations.

## Output format

The script finishes with a block like:

```text
Linear MPC benchmark
params: {'horizon': 18, 'position_weight': 8.0, 'velocity_weight': 1.5, 'control_weight': 0.15, 'delta_u_weight': 0.35, 'terminal_multiplier': 3.0}
---
objective: 12.738057
tracking_cost: 12.355720
control_cost: 0.062997
slew_cost: 0.024488
terminal_cost: 0.001368
constraint_violation: 0.000587
```

Extract the key metric with:

```bash
grep "^objective:" run.log
```

If needed, inspect all reported metrics with:

```bash
grep "^objective:\|^tracking_cost:\|^control_cost:\|^slew_cost:\|^terminal_cost:\|^constraint_violation:" run.log
```

## Logging results

Log each run to `results_linear_mpc.tsv` as tab-separated values:

```text
commit	objective	status	description
```

- `commit`: short git commit hash
- `objective`: use the printed scalar objective; use `0.000000` only for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the attempted change

Do not commit `results_linear_mpc.tsv`.

## Experiment loop

LOOP FOREVER:

1. Check the current git state and identify the current best commit.
2. Modify `linear_mpc_problem.py` with one clear idea.
3. Commit the change.
4. Run the benchmark: `uv run linear_mpc_problem.py > run.log 2>&1`
5. Read `objective` from the log.
6. If the run crashes, inspect the tail of the log, decide whether to fix or discard.
7. Record the result in `results_linear_mpc.tsv` without committing that TSV.
8. If `objective` improved, keep the commit.
9. Otherwise reset back to the previous best commit.

## Search strategy

Start with the exposed scalar knobs in `MPCParams`:
- horizon
- terminal multiplier
- relative scaling between position and velocity weights
- relative scaling between control effort and control variation

Then, if gains plateau, explore benchmark-preserving refinements inside
`linear_mpc_problem.py`, such as:
- scenario set composition
- constraint penalty scale
- rollout length
- structure of the tracking cost

Keep changes interpretable. Prefer one-factor-at-a-time experiments early on.
