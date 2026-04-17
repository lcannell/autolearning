# autoresearch-linear-mpc-3param

This is an experiment to have the LLM do autonomous research on a simplified
linear MPC benchmark with only 3 tunable controller hyperparameters.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: The repo is small. Read these files for context:
   - `README.md` — repository context.
   - `linear_mpc_problem_3param.py` — the single benchmark file you modify.
4. **Initialize results**: Ensure `results_linear_mpc_3param.tsv` exists with just the header row if it is missing.
5. **Confirm and go**: confirm the setup looks correct before starting the search loop.

Once setup is confirmed, start iterating.

## Experimentation

Each experiment is a direct hyperparameter search over a deterministic linear
MPC benchmark. Launch a run with:

```bash
uv run linear_mpc_problem_3param.py > run.log 2>&1
```

The script evaluates a constrained linear tracking controller in closed loop on
a deterministic search split plus a larger deterministic evaluation split, and
prints scalar objectives for both.

**What you CAN do:**
- Modify `linear_mpc_problem_3param.py` only.
- Tune the exposed MPC hyperparameters:
  - `horizon`
  - `control_weight`
  - `delta_u_weight`
- Refine the benchmark internals only if it preserves the intended scope:
  deterministic linear MPC, constrained closed-loop evaluation, and a single
  scalar objective for comparison.

**What you CANNOT do:**
- Add dependencies.
- Spread the benchmark across multiple files unless the user explicitly asks.
- Change the benchmark in ways that destroy comparability across runs.
- Reintroduce the other 3 controller weights into the search space.

## Goal

**The goal is simple: get the lowest `objective`.** Lower is better.

Treat this as a controller-design search problem, not a training problem. Since
the benchmark is deterministic, small improvements are meaningful. The search
metric is `objective`, computed on 32 fixed search scenarios. Also monitor
`eval_objective`, computed on 256 fixed holdout scenarios with a different
fixed seed.

Secondary metrics:
- `eval_objective`
- `search_tracking_cost`
- `search_control_cost`
- `search_slew_cost`
- `search_terminal_cost`
- `search_constraint_violation`
- `eval_tracking_cost`
- `eval_control_cost`
- `eval_slew_cost`
- `eval_terminal_cost`
- `eval_constraint_violation`

Prefer solutions that improve `objective` without exploding control effort or
constraint violations. Prefer configurations that improve both search and
holdout metrics, or that improve `objective` while keeping `eval_objective`
essentially aligned.

## Output format

The script finishes with a block like:

```text
Linear MPC benchmark (3 hyperparameters)
params: {'horizon': 8, 'control_weight': 0.26, 'delta_u_weight': 0.6}
fixed_design: position_weight=7.0, velocity_weight=2.2, terminal_multiplier=1.8
scenario_config: search=32 seed=1234, eval=256 seed=5678
---
objective: 7.059855
eval_objective: 7.562845
search_tracking_cost: 6.645198
search_control_cost: 0.033271
search_slew_cost: 0.014400
search_terminal_cost: 0.004820
search_constraint_violation: 0.000724
eval_tracking_cost: 7.018947
eval_control_cost: 0.036424
eval_slew_cost: 0.015991
eval_terminal_cost: 0.006825
eval_constraint_violation: 0.000969
```

Extract the key metrics with:

```bash
grep "^objective:\|^eval_objective:" run.log
```

If needed, inspect all reported metrics with:

```bash
grep "^objective:\|^eval_objective:\|^search_.*:\|^eval_.*:" run.log
```

## Logging results

Log each run to `results_linear_mpc_3param.tsv` as tab-separated values:

```text
commit	objective	eval_objective	status	description
```

- `commit`: short git commit hash
- `objective`: search metric on the 32-scenario split; use `0.000000` only for crashes
- `eval_objective`: holdout metric on the 256-scenario split; use `0.000000` only for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the attempted change

Do not commit `results_linear_mpc_3param.tsv`.

## Experiment loop

LOOP FOREVER:

1. Check the current git state and identify the current best commit.
2. Modify `linear_mpc_problem_3param.py` with one clear idea.
3. Commit the change.
4. Run the benchmark: `uv run linear_mpc_problem_3param.py > run.log 2>&1`
5. Read both `objective` and `eval_objective` from the log.
6. If the run crashes, inspect the tail of the log, decide whether to fix or discard.
7. Record the result in `results_linear_mpc_3param.tsv` without committing that TSV.
8. Keep the commit if `objective` improved and `eval_objective` does not show a suspicious regression.
9. Otherwise reset back to the previous best commit.
