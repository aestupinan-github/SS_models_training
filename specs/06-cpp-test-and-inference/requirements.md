# Spec 06 — C++ test and inference

**Status:** Deferred but bootstrap-able. Wait until Python pipeline is validated, but design so it can be used early to test individual functionalities.

## Requirements

### R06.1 Standalone

- No dependency on any external DEM or simulation source code.
- Uses system LibTorch (portable; ask for system-wide install or fetch at build time).

### R06.2 Flexible CLI

- Accepts model path and data path as arguments.
- Example: `./test_surrogate --model overlap_model.pt --data test_cases.csv`.

### R06.3 Python + bash wrappers

- Python scripts for plotting/analysis.
- Bash wrappers for execution.
- Reference scripts for style: `inspired_codes/CPP_test/` (partly unfinished, to be reviewed).

### R06.4 CTest integration

- CMake + CTest so tests can be run via `ctest` from the build directory.

### R06.5 Inference pipeline

1. Load TorchScript model.
2. Load ASCII scalers (`scalers.dat`).
3. Preprocess input (quaternion canonicalization).
4. Run inference.
5. Postprocess (inverse scaler, `10^pred - EPS`).
6. Output predictions CSV.

### R06.6 Cross-check

- Compare C++ predictions against Python surrogate and analytical reference.

### R06.7 Test cases

- Reuse the fixed test cases from spec 04 if not heavy.

### R06.8 Portability

- Do not hardcode `Programs/libtorch`.
- Use CMake to find LibTorch or fetch it at build time.
