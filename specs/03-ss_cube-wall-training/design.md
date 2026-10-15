# Spec 03 — Design

## Status

**DRAFT, revision 2, 2026-10-04. Not approved.** Follows `requirements.md`
revision 2 (approved 2026-10-04). Supersedes design revision 1 (git history).

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
| `interactions/ss_cube-wall/python/surrogate_model.py` | `UFeature`, `HNet`, `SurrogateModel`, status codes, export | New |
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
h_pred = HNet(u)                         (u scaled inside HNet if a scaling is used)
sd     = position_z - h_pred
far    = position_z > Z_PREFILTER        ; sd[far] = position_z - R_CIRC
status = in_range; too_deep where sd < -0.1 (not far); far; invalid where not valid (sd = NaN)
```

Constants `R_CIRC`, `Z_PREFILTER`, the `-0.1` bound are stored as module
attributes, taken from `cube_wall` at build time, and recorded in the run
record. The network is evaluated for every row (vectorized); `far` rows
overwrite the result. This meets R03.4 "SHALL NOT need the network" because
the result for `far` rows does not depend on it.

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
  end; sha256 of `model.pt` and `state_dict.pth`; label "SMOKE" when within
  smoke limits.

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
