# Spec 03 — Design

## Target transform

The signed distance spans `[-0.5, +0.1]` and is symmetric in magnitude but signed. Training in log space compresses the range and gives equal relative weight to small and large values.

The signed log transform is:

```
t(x) = sign(x) * log10(|x| + EPS)
```

with `EPS = 1e-15`. This maps:
- `t(0) = 0`
- small penetration `x = -1e-5` → `t ≈ 5`
- large penetration `x = -0.5` → `t ≈ -0.3`

The inverse is:

```
x = sign(t) * (10^|t| - EPS)
```

## Model

SIREN MLP with sine activations. Prior work used 256→128→64→32→16 hidden layers. This is a starting point; architecture is open to exploration.

## Training loop

1. Load CSV, extract features `[position_z, qw, qx, qy, qz]` and target `signed_distance`.
2. Canonicalize quaternions (`qw >= 0`).
3. Transform target: `sign(x) * log10(|x| + EPS)`.
4. Split 80/20 with fixed seed.
5. Fit scalers on train only.
6. Train with MSE loss in transformed space.
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
