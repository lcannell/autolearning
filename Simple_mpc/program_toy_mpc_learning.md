# Purpose
We need to calibrate the four parameters of an MPC optimization problem through autonomous learning.

These parameters belong to the MPC cost function, but the cost function itself is unknown to us.

# Methodology
We can only observe a performance functional called “objective,” which penalizes how well the controller tracks a given reference.

Your task is to iteratively tune the parameters by testing different configurations, observing the resulting objective output, and using that feedback to guide future attempts.

In essence, you must solve a black-box optimization problem.


# Instructions
The main code you must run is `toy_mpc_qp.py`.
All utilities are in `toy_mpc_qp_utils.py`.
You are NOT allowed to modify `toy_mpc_qp_utils.py`; you may only modify `toy_mpc_qp.py`.
Run experiments continuously without asking the user for feedback. Keep going autonomously and indefinitely.


Create a branch named using the month-day format, for example `apr04` if today were April 4.
Save the results in `results_mpc.tsv` — tab-separated, not comma-separated, because commas break descriptions. The TSV must have a header row and 5 columns:
commit	objective	status	description	notes

1. Git commit hash, short format, 7 characters.
2. Objective achieved, for example 1.234567; use 0.000000 for crashes.
3. Status: keep, discard, or crash.
4. Short text description of what this experiment tried.
5. Optional notes, such as the reasoning behind the choice of the values

The first thing you must do is to run a benchmark by selecting 4 random values for the hyperparameters, then you start the black-box search
