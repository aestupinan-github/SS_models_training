# Spec 03 — ss_cube-wall training

## Requirements

### R03.1 Model architecture

- SIREN-style MLP (sine activations).
- 5 inputs → hidden layers → 1 output.
- Hidden layer sizes open to exploration (prior work: 256→128→64→32→16).
- No new architecture; reuse and adapt existing.

### R03.2 Target transform

- Signed log transform: `t(x) = sign(x) * log10(|x| + EPS)` with `EPS = 1e-15`.
- Train in transformed space.
- Recover signed distance via `x = sign(t) * (10^|t| - EPS)`.

### R03.3 Scalers

- `StandardScaler` on inputs (fit on train only).
- `StandardScaler` on transformed target (fit on train only).
- Export as:
  - joblib `.save` files (Python)
  - ASCII `scalers.dat` (for C++ side: `x_mean`, `x_scale`, `y_mean`, `y_scale`, `eps`)

### R03.4 Loss

- MSE in transformed space (single-output regression).

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

- Primary: RMSE in transformed (signed-log) space.
- Secondary: absolute and relative error in linear space, reported on the penetration subset for interpretability.
