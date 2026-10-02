# Spec 03 — Design

## Target transform

The overlap spans `[1e-5, 0.5]` — over 4 orders of magnitude. Training in log space (`log10(overlap + EPS)`) compresses this range and gives equal relative weight to small and large overlaps.

## Model

SIREN MLP with sine activations. Prior work used 256→128→64→32→16 hidden layers. This is a starting point; architecture is open to exploration.

## Training loop

1. Load CSV, extract features `[position_z, qw, qx, qy, qz]` and target `overlap`.
2. Canonicalize quaternions (`qw >= 0`).
3. Transform target: `log10(overlap + EPS)`.
4. Split 80/20 with fixed seed.
5. Fit scalers on train only.
6. Train with MSE loss in log space.
7. Early stopping on validation loss.
8. Export TorchScript + state_dict + scalers.

## Scaler export

ASCII `scalers.dat` format (for C++ consumption):

```
x_mean
<v0> <v1> <v2> <v3> <v4>
x_scale
<s0> <s1> <s2> <s3> <s4>
y_mean
<value>
y_scale
<value>
eps
<value>
```
