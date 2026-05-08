# autoresearch

This is an experiment where the LLM improves a toy MPC benchmark by turning `toy_mpc_qp.py` into an automatic search loop over the hyperparameters in `MPCParams`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `toy_mpc_qp_utils.py` — data prep, optimizer, evaluation. Do not modify.
   - `toy_mpc_qp.py` — the file you modify. Hyperparameter search procedure.
4. **Initialize results_mpc.tsv**: Create `results_mpc.tsv` with just the header row. The baseline will be recorded after the first run.


## Experimentation

Each experiment evaluates candidate MPC hyperparameters on a **fixed budget of scenarios**. You launch it simply as: `uv run toy_mpc_qp.py`.

**What you CAN do:**
- Modify `toy_mpc_qp.py` — this is the only file you edit.
- Add code that performs repeated attempts to find the best combination of the hyperparameters in `MPCParams`.
- Choose search strategy that an human could not simply implement: avoid for loops and simple grid searches.
- Add useful per-attempt logging so each trial prints the most important metrics.
- You CAN and you are invited to install new packages or add dependencies, BUT first ask to the user.


**What you CAN NOT do:**
- Modify `toy_mpc_qp_utils.py`. It is read-only. It contains the fixed evaluation setup, scenario generation, and benchmark logic.
- Modify the evaluation procedure. The `objective` in `toy_mpc_qp_utils.py` is the ground truth metric.

**The goal is simple: get the lowest objective** You should modify `toy_mpc_qp.py` so it automatically explores the hyperparameters and returns the best configuration it found.

At each attempt, the script should log a concise but informative summary. At minimum, include:

- attempt number
- hyperparameters tested
- objective
- value of the terms in the objective
- whether the result is the current best

At the end of the run, the script should print the best configuration found and the associated final metrics.

**The first run**: Your very first run should establish the current baseline behavior of `toy_mpc_qp.py` before you make the search loop more sophisticated.



## Output format
The exact formatting is up to you, but the output should include per-attempt logs plus a final best-result summary. A reasonable attempt log might look like this:

```
attempt=7 best_so_far=yes params={...}
objective=5.812300 t_raise=5.780000 settling_time=0.020100 t_overshoot=0.007200 constraint_violation=0.000000
```

And the final summary should clearly report the best configuration and its metrics.

You can still extract the key metric from the log file with:

```
grep "^objective:" run.log
```

## Logging results

When an experiment is done, log it to `results_mpc.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	objective	status	description	 notes
```

1. git commit hash (short, 7 chars)
2. objective achieved (e.g. 1.234567) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried
5. optional notes, such as the search strategy or the best parameter set found

Example:

```
commit	objective	status	description	notes
a1b2c3d	6.000022	keep	baseline	static MPCParams
b2c3d4e	5.812300	keep	random search over MPCParams	best params after 40 attempts
c3d4e5f	6.104200	discard	local search around best point	no improvement
d4e5f6g	0.000000	crash	bayesian search prototype	bug in candidate generation
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `toy_mpc_qp.py` with an experimental idea by directly hacking the search logic.
3. git commit
4. Run the experiment: `uv run toy_mpc_qp.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results from the final best-result summary in `run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results_mpc.tsv file, leave it untracked by git)
8. If objective improved (lower), you "advance" the branch, keeping the git commit
9. If objective is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they do not, discard. And you are advancing the branch so that you can iterate. If you feel like you are getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The user then wakes up to experimental results, all completed by you while they slept!
