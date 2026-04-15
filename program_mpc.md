# autoresearch-mpc

This is an experiment to have the LLM do autonomous research on a neural policy
that imitates an MPC controller.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train_mpc.py` - the file you modify. Model architecture, optimizer, training loop.
4. **Initialize results.tsv**: Create `results_mpc.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**:C onfirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 1 minute**. Each experiment runs by launching:

```bash
uv run train_mpc.py
```

The script trains a neural network policy to imitate a finite-horizon MPC/LQR
teacher and then evaluates the learned controller in closed loop.

**What you CAN do:**
- Modify `train_mpc.py` only. This is the only file you edit.
- Tune architecture, optimizer, dataset generation, rollout horizon, and other
  hyperparameters.
- In particular, explore neural architecture parameters such as:
  - `NUM_LAYERS`
  - `HIDDEN_SIZE`
  - `ACTIVATION`

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the scope of the benchmark outside `train_mpc.py`.

## Goal
**The goal is simple: get the lowest `eval_cost`.** The only constraint is that the code runs without crashing. Since the time budget is fixed, you don't need to worry about training time — it's always 1 minute.


**Simplicity criterion**: Lower is better. If two runs are very close, prefer the simpler model.
A modest reduction in `num_layers` or `hidden_size` with equal performance is a
real win.

Secondary metrics:
- `teacher_mse`
- `state_rms`
- `peak_vram_mb`

## Output format

The script finishes with a block like:

```text
---
eval_cost:        12.345678
teacher_mse:      0.012345
state_rms:        0.456789
best_val_mse:     0.001234
training_seconds: 60.0
peak_vram_mb:     123.4
num_params:       34567
num_layers:       3
hidden_size:      128
```

Extract the key metrics with:

```bash
grep "^eval_cost:\|^peak_vram_mb:" run.log
```

## Logging results

Log each run to `results_mpc.tsv` as tab-separated values:

```
commit	eval_cost	memory_gb	status	description
```

- `commit`: short git commit hash
- `eval_cost`: use `0.000000` only for crashes
- `memory_gb`: `peak_vram_mb / 1024`, rounded to one decimal
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the attempted change

## Experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `train_mpc.py` with one clear idea.
3. git commit.
4. Run the experiment: `uv run train_mpc.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read `eval_cost` and `peak_vram_mb` from the log.
6. If the run crashes, inspect the tail of the log, decide whether to fix or discard.
7. Record the result in `results_mpc.tsv` without committing that TSV.
8. If `eval_cost` improved, keep the commit.
9. Otherwise reset back to the previous best commit.

Primary search axis at the beginning:
- number of hidden layers
- hidden width

Then explore:
- activation function
- dataset size
- rollout/evaluation settings
- optimizer and regularization
