# Spec 03 — Design

## Status

**DRAFT, revision 2, 2026-10-04. Not approved.** Follows `requirements.md`
revision 2 (approved 2026-10-04, amended the same day for `scalers.dat`,
S03.10). Supersedes design revision 1 (git history).

### Decisions

- **D03.1 `scalers.dat` layout (agent, delegated by the user 2026-10-04:
  "I'll let you decide"; revised 2026-10-04, S03.11).** One `key value...`
  record per line, whitespace-separated, `#` starts a comment line; parsed
  in C++ with `std::getline` and `std::istringstream` only. Keys are
  unique except `ref`, which repeats (one reference pair per line).
  Contents: format version, run id, `model.pt` sha256, input and output
  definition, status codes, `y_mean` and `y_scale` of `h`, constants, and
  the reference input/output pairs. Reason: easy to read and to check by
  hand, no parser library needed, and the reference pairs let C++ verify
  column order, quaternion order, dtype and model identity at load time.
  *Superseded first version (same day):* the layout of the old reference
  C++ loader with an `eps` key; dropped because no compatibility with the
  old code is wanted (user: the old model is only a reference). Rejected
  alternative: JSON, which needs a parser library in C++.

## Requirements traceability

| Design element | Requirements |
|---|---|
| `UFeature` fixed input layer | R03.3 |
| `HNet` trainable network | R03.1, R03.14 |
| `SurrogateModel` exported module (sd, status) | R03.1, R03.2, R03.3, R03.4, R03.9 |
| Dataset loader with checksum | R03.5, R03.13 |
| Orientation split | R03.6 |
| Loss and training loop | R03.7, R03.10, R03.12, R03.13, R03.14 |
| Metrics, incl. tumbling path | R03.8 |
| Run writer and run record | R03.11, R03.13 |
| `scalers.dat` writer (D03.1) | R03.16 |
| `train()` entry point and CLI | R03.15 |

## Existing code context

- Spec 01: `cube_wall.py` (reference `h`, constants `R_CIRCUMSCRIBED`,
  `Z_PREFILTER`, `SD_COVER_MAX`, `classify_zone`).
- Spec 02: `generate_dataset.py` (`data.npz` with columns `position_z, qw,
  qx, qy, qz, sd, h, orientation_id, component, sd_part, delta`;
  `provenance.json` with `sha256`; codes `COMPONENT_CODES`, `SD_PART_CODES`).
  Its writer pattern (temporary folder, safe publish, never overwrite) is
  reused for run folders.
- Environment (*Confirmed*, 2026-10-04): PyTorch 2.12.1+cu130, NVIDIA RTX
  A4000 usable (an NVML warning is printed and harmless), NumPy 2.3.1. No new
  dependency.

## Feasibility measurements (2026-10-04, probe code, not results)

- TorchScript `script` + `save` + `load` of the module below works; status
  codes `in_range`, `far`, `invalid` returned as designed.
- `q` and `-q` gave bit-identical `sd` on 1e5 random quaternions.
- Float32 network vs the same weights in float64: largest `h` difference
  5.2e-7.
- A 20 s CPU fit (MLP 3-64-64-1, SiLU) on 1000 orientations of
  `dataset_001` reached a validation `|h|` error median 6.5e-3, max 4.2e-2.
  This only shows the pipeline learns; it is far from any useful accuracy and
  is not a result.

## Files and responsibilities

| File | Responsibility | Change |
|---|---|---|
| `interactions/ss_cube-wall/python/surrogate_model.py` | `UFeature`, `HNet`, `SurrogateModel`, status codes, export, `scalers.dat` writer and reader (`write_model_record`, `read_model_record`) | New |
| `interactions/ss_cube-wall/python/train_surrogate.py` | Config, loader, split, loss, loop, metrics, run writer, `train()`, CLI | New |
| `tests/test_surrogate_model.py` | Model and interface tests | New |
| `tests/test_train_surrogate.py` | Training pipeline tests | New |
| `.gitignore` | Ignore `interactions/*/output/run_*/` and `.tmp_run_*/` | Extended (side effect recorded in MEMORY) |
| `interactions/ss_cube-wall/README.md` | Run listing | Updated per run |

The model code is separate from the training code so the exported module has
no dependency on the training loop.

## Interfaces

### Exported model (R03.2)

```
SurrogateModel.forward(x: Tensor[N, 5] float64) -> (sd: Tensor[N] float64,
                                                    status: Tensor[N] int64)
columns of x: position_z, qw, qx, qy, qz          (canonical units, L = 1)
STATUS_IN_RANGE = 0, STATUS_FAR = 1, STATUS_TOO_DEEP = 2, STATUS_INVALID = 3
```

Algorithm (all float64):

```
valid  = all(isfinite(x), axis=1) and |q| > 0
q      = q / |q|                         (rows not valid: replaced by (1,0,0,0) for computation only)
u      = -(third row of R(q))            = -(2(xz - wy), 2(yz + wx), 1 - 2(x^2 + y^2))
h_pred = HNet(u) = y_scale * MLP(u) + y_mean     (y_mean, y_scale fixed buffers)
sd     = position_z - h_pred
far    = position_z > Z_PREFILTER        ; sd[far] = position_z - R_CIRC
status = in_range; too_deep where sd < -0.1 (not far); far; invalid where not valid (sd = NaN)
```

Constants `R_CIRC`, `Z_PREFILTER`, the `-0.1` bound are stored as module
attributes, taken from `cube_wall` at build time, and recorded in the run
record and in `scalers.dat`.

The network is evaluated for every row (vectorized); `far` rows overwrite
the result. This meets R03.4 "SHALL NOT need the network" because the
result for `far` rows does not depend on it.

### `scalers.dat` (R03.16, D03.1)

Output scaling: `y_mean`, `y_scale` are the mean and standard deviation of
`h` over the training orientations (training split only), fixed before
training and stored as `HNet` buffers. The MLP learns the scaled target
`(h - y_mean) / y_scale`; `HNet` undoes it, so the model output is final.
No input scaling: `u` is a unit vector.

```
# ss_cube-wall surrogate: plain-text record of model.pt (spec 03 R03.16)
# The model output sd is final. Do not apply these values to it.
format ss_model_record_v1
run_id run_001
model_sha256 <64 hex>
inputs position_z qw qx qy qz
input_dtype float64
input_units canonical_L1
outputs sd status
output_dtype float64 int64
status_codes 0:in_range 1:far 2:too_deep 3:invalid
y_mean <17 significant digits>
y_scale <17 significant digits>
r_circ <17 significant digits>
z_prefilter <17 significant digits>
sd_too_deep -0.10000000000000001
ref_eval one_row_per_call_float64_cpu
ref_sd_tol 1e-14
n_ref <n>
# ref columns: position_z qw qx qy qz sd status   (%.17g; sd nan for invalid)
# sd is bitwise for one row per call; within ref_sd_tol for any batch size
ref 0.69999999999999996 1 0 0 0 0.19... 0
ref ...
```

Reference inputs (fixed, 17 digits): face-down, edge-down and corner-down
orientations at heights giving gap, contact and overlap; one random
orientation and its `-q`; a non-unit quaternion (`2q`); a `far` row
(`z > z_prefilter`); a `too_deep` row (deep overlap); an `invalid` row
(zero quaternion). The outputs are computed by the exported float64 model
on CPU, one row per call, when the file is written. Measured 2026-10-04:
text round trip and save/load are exact; batched evaluation (batch >= 20)
and thread count change the last bit (up to 4.4e-16), hence `ref_sd_tol`.

`write_model_record(model, path, run_id, model_sha256)` writes it;
`read_model_record(path)` parses it (unique keys to a dict, `ref` lines to a
list) and is used by the tests. Constraint recorded for spec 06: C++ uses
`sd` directly; it may use `scalers.dat` to verify the loaded model by
running the reference inputs.

`u` is quadratic in `q`, so `q -> -q` gives identical `u` bit for bit
(R03.3, measured).

### Training API (R03.15)

```
@dataclass(frozen=True)
class TrainConfig:
    dataset: str                     # path to a spec 02 dataset folder
    seed: int
    val_fraction: float = 0.2
    hidden: tuple = (64, 64, 64)
    activation: str = "silu"         # "silu" | "tanh" | "sine"
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 2000
    patience: int = 200              # epochs without validation improvement
    loss: str = "mse"                # on h, canonical units
    device: str = "auto"             # "cpu" | "cuda" | "auto"
    max_wall_seconds: float = 600    # R03.12 smoke limit
    allow_full_scale: bool = False   # > 1e4 rows or > 600 s

validate_train_config(cfg) -> None       # raises TrainConfigError
load_dataset(path) -> (arrays, provenance)   # verifies sha256; raises DatasetError
split_by_orientation(orientation_id, fraction, seed) -> (train_ids, val_ids)
train(cfg, out_root) -> (run_path, metrics)
```

CLI: `python train_surrogate.py --dataset DIR --seed S [--hidden 64,64,64]
[--epochs N] [--device cpu|cuda] [--out DIR] [--allow-full-scale]`.
Exit codes as in spec 02: 0 written, 2 invalid input, 1 failure.

## Data

- Training targets are per orientation: `h` is identical for all `K` rows
  of an orientation (*Confirmed*, dataset_001), so training uses one sample
  per `orientation_id` (input `u(q)`, target `h`). `K` rows matter only for
  `sd` metrics.
- Split by `orientation_id` with a seeded permutation; both sets recorded by
  their sha256 of the sorted id list.

## Numerical method

- **Loss (R03.7):** MSE on `h_pred - h`, canonical units, unweighted, as a
  starting point. Alternatives (relative weighting near kinks, L1, max-error
  terms) are design options for later runs, chosen per run and recorded.
- **Network (R03.14 starting values):** MLP `3 -> 64 -> 64 -> 64 -> 1`,
  SiLU, Adam lr `1e-3`, batch 256, early stopping with patience 200 on the
  validation MSE, best weights kept. SIREN (sine) is selectable for
  comparison. These are starting values, not requirements.
- **Precision (R03.9):** training float32 on the chosen device; export by
  copying the best weights into a float64 `SurrogateModel`. The float32 vs
  float64 difference on the validation set is a metric (R03.8).
- **Reproducibility (R03.10):** seeds for NumPy, `torch.manual_seed`, split,
  batch order (a `torch.Generator`); `torch.use_deterministic_algorithms(True)`
  and, on CUDA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. CPU runs: bitwise
  identical weights. GPU runs: final validation metrics equal within relative
  `1e-6` (stated tolerance).

## Metrics (R03.8)

Computed with the float64 exported model on the validation orientations
and their rows:

- `h`: max, p99, median, RMS of `|h_pred - h|`; per component; per decade of
  `Δ` (`[1e-6, 1e-1]` and `Δ = 0` exactly, i.e. special rows).
- `sd`: same statistics per zone (`accuracy`, `gap_safety` via
  `cube_wall.classify_zone` of the exact `sd`) and per `|sd|` decade and
  sign.
- Sign safety at `DELTA_SIGN` in `{0, 1e-5, 1e-4, 1e-3}`: false-contact rate
  (exact `sd > DELTA_SIGN`, predicted `< 0`), missed-contact rate (exact
  `sd < -DELTA_SIGN`, predicted `>= 0`), largest spurious overlap.
- Continuity: largest `|sd(q) - sd(-q)|` over validation rows (expected 0);
  tumbling path: a fixed seeded path of 20000 steps of about 1 mrad, at a
  fixed height crossing contact; largest `|Δsd_pred - Δsd_exact|` between
  steps and largest `|sd_pred - sd_exact|` on the path.
- Precision: largest `|h_f32 - h_f64|`.
- Status counts on validation rows.

No pass/fail thresholds (tolerances open until the baseline).

## Run writer (R03.11, R03.13)

- `out_root = interactions/ss_cube-wall/output`; new `run_XXX`, counter =
  1 + largest existing; built in `.tmp_run_XXX_<pid>` and published with the
  same safe publish as spec 02 (D02.6): never overwrites, even an empty
  folder.
- Contents:

```
run_XXX/
  model.pt          # TorchScript SurrogateModel, float64
  state_dict.pth    # HNet weights (float32 training copy)
  scalers.dat       # plain-text copy of scaling and constants (R03.16, D03.1)
  run.json          # record (below)
```

The flat layout replaces the `models/ configs/ logs/ results/` subfolders
of spec 04 revision 1, which is still to be revised (side effect recorded in
MEMORY).

- `run.json`: spec revisions; script names; git commit and dirty flag;
  Python, NumPy, PyTorch versions; device name; `TrainConfig`; seeds;
  dataset path, its `data.npz` sha256 and its provenance seed; split id
  hashes; model constants; parameter count; loss curve (train and
  validation per epoch); epochs run and stop reason; metrics; UTC start and
  end; sha256 of `model.pt`, `state_dict.pth` and `scalers.dat`; label
  "SMOKE" when within smoke limits.

## Error handling

- `TrainConfigError(ValueError)` before anything is written.
- `DatasetError(ValueError)`: missing folder or files, checksum mismatch,
  missing columns.
- `TrainingError(RuntimeError)`: non-finite loss, export or write failure;
  the temporary folder is left and named in the message.
- Smoke limit: more than `1e4` rows or `max_wall_seconds > 600` without
  `allow_full_scale` raises `TrainConfigError`. During training, reaching
  `max_wall_seconds` stops training (stop reason recorded), not an error.

## Testing and validation strategy

| Test | Covers |
|---|---|
| `u(q)` equals `-R(q)^T e_z` from scipy; bit-identical for `q`, `-q` | R03.3 |
| Exported model: shapes, dtypes, TorchScript save/load, no Python-side file | R03.2 |
| Status: `far` above `Z_PREFILTER` with `sd = z - R_CIRC`; `too_deep`; `invalid` with NaN; mixed batch; batch of one; `z = Z_PREFILTER` is `in_range` | R03.4 |
| `sd = z - h_pred` exactly (slope 1) | R03.1 |
| Float64 export equals float32 network within 1e-6 | R03.9 |
| `scalers.dat`: scaling and constants equal the model buffers exactly; `model_sha256` matches `model.pt`; running the loaded `model.pt` on every `ref` input reproduces its status exactly and its `sd` bit for bit one row per call, and within `ref_sd_tol` as one batch (CPU, float64); every status code covered; 17-digit round trip; parse with whitespace splitting only | R03.16 |
| Loader: checksum mismatch, missing file, missing column raise `DatasetError` | R03.5, R03.13 |
| Split: disjoint orientation sets, fraction, reproducible | R03.6 |
| Config validation and smoke limits | R03.12, R03.14 |
| Training on a tiny synthetic dataset reduces the loss; non-finite loss raises | R03.7, R03.13 |
| CPU run twice: identical weights and metrics | R03.10 |
| Metrics: known values on a constructed prediction | R03.8 |
| Run writer: new `run_XXX`, never overwrites, `run.json` fields, checksums | R03.11 |
| `train()` returns path and metrics; CLI exit codes | R03.15 |
| Smoke run on `dataset_001` (completion criterion, reported as smoke) | all |

Tests use temporary folders and tiny datasets; only the final smoke run
writes to `output/`.

## Risks and limitations

- `h` has kinks (faces, edges); a smooth MLP cannot represent them exactly,
  so the largest errors are expected near kinks (measured per `Δ` decade).
- `dataset_001` has only 1250 orientations; the smoke run shows the pipeline
  works, not the achievable accuracy (that is spec 07).
- GPU determinism depends on library support; the stated tolerance covers
  remaining differences.
- `too_deep` from predicted `sd` only (R03.4 limitation).
