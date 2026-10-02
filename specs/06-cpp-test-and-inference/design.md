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
8. Inverse scale output: `log_overlap = y_pred * y_scale + y_mean`.
9. Recover overlap: `overlap = 10^log_overlap - eps`.
10. Write predictions CSV.

## CTest integration

`CMakeLists.txt` defines test cases that run the binary with different inputs and check outputs.

## Inspiration

Inspired by `XDEM/Cases/CPP_test/` but independent fresh development.
