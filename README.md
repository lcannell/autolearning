# autoresearch

The idea is to give an AI agent a small but real MPC tuning problem and let it experiment autonomously. The agent edits `toy_mpc_qp.py`, tries different strategies to choose the best combination of the 5 MPC hyperparameters, runs the benchmark on a fixed set of scenarios, checks whether the objective improved, and keeps iterating. The human does not manually tune the controller in code; instead, the human edits `program_toy_mpc.md` to define the agent's operating instructions and research style.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`toy_mpc_qp_utils.py`** — utilities for the MPC benchmark, fixed constants, scenario generation, simulation, and objective computation.
- **`toy_mpc_qp.py`** — the single file the agent edits. It should contain the hyperparameter search logic. **This file is edited and iterated on by the agent**.
- **`program_toy_mpc.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, evaluation runs on a fixed set of **24 scenarios**. The main metric is **objective** — lower is better.

## What The Agent Should Do

The purpose of `toy_mpc_qp.py` is to search over the 5 hyperparameters in `MPCParams` and find a combination that minimizes the objective returned by the benchmark.

The agent is free to choose the search strategy. For example, it can implement:

- random search
- local search
- Bayesian optimization
- population or swarm-based search
- any hybrid strategy it judges useful

The important point is not which optimizer is used, but that `toy_mpc_qp.py` performs repeated attempts and searches for a better hyperparameter configuration automatically.

For each attempt, the script should print a compact log with the most important metrics so the experiment is easy to inspect while it runs. At minimum, each attempt log should make it clear:

- which hyperparameters were tried
- what objective was obtained
- the main component metrics such as tracking cost, input cost, input-rate cost, and constraint violation
- whether this attempt is the best one so far

At the end of the run, the script should print the best hyperparameter configuration found and its associated benchmark metrics.

## Quick start

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/). A GPU can be used if available, but the code can also run on CPU.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Manually run a single training experiment
uv run toy_mpc_qp.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Read program_toy_mpc.md and turn toy_mpc_qp.py into a hyperparameter search loop that finds a better set of MPCParams and logs the important metrics for every attempt.
```

The `program_toy_mpc.md` file is essentially a super lightweight "skill".


## Project structure

```
toy_mpc_qp_utils.py      — constants, data prep + runtime utilities (do not modify)
toy_mpc_qp.py            — hyperparameter search loop (agent modifies this)
program_toy_mpc.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `toy_mpc_qp.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed evaluation budget.** The benchmark always runs over the same 24 scenarios. This keeps experiments directly comparable across different search strategies.
- **Search method is flexible.** The agent is allowed to choose whatever search method it prefers for the 5 MPC hyperparameters, as long as it improves the objective.
- **Per-attempt visibility.** Every attempted hyperparameter configuration should be logged with the key metrics so that progress is inspectable and reproducible.
- **Self-contained.** No distributed setup, no complex config system. One benchmark, one file to iterate on, one main metric to optimize.
