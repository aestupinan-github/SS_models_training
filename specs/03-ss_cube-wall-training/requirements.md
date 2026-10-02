# Spec 03 — ss_cube-wall training

## Requirements

### R03.1 Model architecture

- SIREN-style MLP (sine activations).
- 5 inputs → hidden layers → 1 output.
- Hidden layer sizes open to exploration (prior work: 256→128→64→32→16).
- No new architecture; reuse and adapt existing.

### R03.2 Target transform

- `log10(overlap + EPS)` with `EPS = 1e-15`.
- Train in log space.
- Recover overlap via `10^pred - EPS`.

### R03.3 Scalers

- `StandardScaler` on inputs (fit on train only).
- `StandardScaler` on log-target (fit on train only).
- Export as:
  - joblib `.save` files (Python)
  - ASCII `scalers.dat` (for C++ side: `x_mean`, `x_scale`, `y_mean`, `y_scale`, `eps`)

### R03.4 Loss

- MSE in log space (single-output regression).

### R03.5 Train/validation split

- 80/20, fixed seed, shuffle.

### R03.6 Optimizer

- AdamW, lr `1e-4`, weight decay `1e-7`.
- ReduceLROnPlateau scheduler.
- Early stopping on validation loss.

### R03.7 Export format

- TorchScript `.pt` + state_dict `.pth` + scalers (joblib + ASCII).

### R03.8 Seeds

- Fixed and recorded (numpy, torch, split).

### R03.9 Validation metrics

- Primary: RMSE in log space (best for small overlaps).
- Secondary: relative error in linear space (human-readable).
