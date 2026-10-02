# Spec 06 — Design

## Build system

CMake with CTest. `CMakeLists.txt` finds LibTorch via `find_package` or fetches it.

## CLI design

```
./test_surrogate --model <path> --data <path> --output <path>
```

## Inference pipeline

1. Parse CLI arguments.
2. Load TorchScript model via `torch::jit::load`.
3. Parse ASCII `scalers.dat` (x_mean, x_scale, y_mean, y_scale, eps).
4. Read input CSV.
5. Canonicalize quaternions (`qw >= 0`).
6. Scale inputs: `x_scaled = (x - x_mean) / x_scale`.
7. Run model.
8. Inverse scale output: `t_scaled = y_pred * y_scale + y_mean`.
9. Recover signed distance: `x = sign(t_scaled) * (10^|t_scaled| - eps)`.
10. Clamp to `[-0.5, 0.1]`.
11. Write predictions CSV.

## CTest integration

`CMakeLists.txt` defines test cases that run the binary with different inputs and check outputs.

## Inspiration

Inspired by the reference scripts in `inspired_codes/CPP_test/` (partly unfinished, to be reviewed) but independent fresh development.
