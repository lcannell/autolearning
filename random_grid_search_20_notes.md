# Random Grid Search 20

Seed: `20260512`

Sampling:
- `prediction_horizon`: uniform integer in `[10, 30]`
- `control_horizon_fraction`: uniform in `[0.1, 1.0]`
- `q_delta_u_v`: log-uniform with `log(q)` in `[-5, 3]`
- `q_delta_u_psi`: log-uniform with `log(q)` in `[-5, 3]`

Best result: row `3`, objective `15.062992`.
