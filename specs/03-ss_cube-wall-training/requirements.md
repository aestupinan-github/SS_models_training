# Spec 03 — ss_cube-wall training

## Status

**Revision 2, 2026-10-04. APPROVED by the user (2026-10-04, "ok").** Supersedes revision 1 (git
history). Built on the decisions approved by the user on 2026-10-04
(S03.1-S03.8 in `MEMORY.md`, "Spec 03 decisions"). R03.4 output conventions
confirmed by the user for a first try (2026-10-04). Amended 2026-10-04
(user change request, option A, decision S03.10): R03.2, R03.11 changed,
R03.16 added for `scalers.dat`; amendment approved by the user ("ok").
Review issue numbers are
cited in brackets. `design.md` and `tasks.md` are not updated until this file
is approved. Items marked *[OPEN]* need a user decision.

## Context and objective

*Confirmed (user).* The surrogate replaces the overlap computation of a new
LIGGGHTS shape type and is evaluated from C++ at runtime. Overlap accuracy
comes first. The cube is a testbed; a fully trained model is the primary
approach. Spec 01 (rev 3) defines the contract; spec 02 delivers the
training datasets.

This spec defines how a surrogate is trained on a spec 02 dataset, what the
exported model computes, and what each run records and reports.

## Scope

- In scope: training pipeline, model and its exported interface, metrics,
  run artifacts, a smoke training run on `dataset_001`.
- Out of scope: the learning-curve study (new spec 07, S03.6); full-scale
  training (after spec 07 and approval); fixed evaluation test sets (spec
  04); C++ inference (spec 06); contact point and normal (deferred, S03.7).

## Definitions

- **`h(q)`:** support height, spec 01 R01.16. `sd = position_z - h(q)`.
- **`u(q)`:** body-frame direction towards the wall, `u = -R(q)^T e_z`. A
  quadratic function of the quaternion components; identical for `q` and
  `-q`.
- **Network:** the trainable part. **Model:** the exported module, network
  plus the fixed pre- and post-processing of this spec.
- **Run:** one training execution. **Artifacts:** the files it writes.
- **Zones:** spec 01 R01.13 (`accuracy`, `gap_safety`, `too_deep`, `beyond`).

## Requirements

### R03.1 Learning target *(S03.1)* [1, 2]

- The network SHALL learn `h(q)`. The model SHALL output
  `sd = position_z - h_pred(q)`, so `d sd / d position_z = 1` exactly.
- No transform of `sd` (log or other) is used.

### R03.2 Model interface *(S03.5)* [8]

- Input: `x` of shape `(N, 5)`, columns `[position_z, qw, qx, qy, qz]`,
  float64, canonical units (spec 01 R01.6). `N >= 1`.
- Output: `sd` of shape `(N,)`, float64, canonical units, and `status` of
  shape `(N,)`, integer (R03.4).
- The model SHALL be a single TorchScript file and need no other file and no
  Python at runtime (constitution 13). Any scaling lives inside it.
- *[changed 2026-10-04, S03.10]* A plain-text `scalers.dat` is written next
  to the model as a record of the scaling and constants used inside it
  (R03.16). It is not needed to compute `sd`; the model output is final.
- Rescaling by the particle size `L` stays with the caller (spec 01 R01.7).

### R03.3 Sign invariance and normalization *(S03.2)* [3]

- The model SHALL normalize `q` and see it only through `u(q)`.
- For any valid input, the outputs for `q` and `-q` SHALL be bit-identical
  (spec 01 R01.18).

### R03.4 Pre-filter and range status [9]

- WHEN canonical `position_z > Z_PREFILTER`, THE MODEL SHALL return status
  `far` and SHALL NOT need the network for that row (spec 01 R01.13).
- WHEN the predicted `sd < -0.1`, THE MODEL SHALL return status `too_deep`
  and the unclamped predicted `sd`.
- IF the quaternion has zero norm or a non-finite component, or
  `position_z` is non-finite, THEN THE MODEL SHALL return status `invalid`.
- Otherwise status `in_range`.
- Status values: `0 in_range`, `1 far`, `2 too_deep`, `3 invalid`.
  *Confirmed (user, 2026-10-04, "for a first try")*; may be revised after the
  first LIGGGHTS trial.
- `sd` returned for `far`: `position_z - R_CIRC`, a guaranteed lower bound of
  the exact `sd`, always above `0.1`, finite. *Confirmed (user, first try).*
  Note: at `Z_PREFILTER` the returned `sd` steps from up to `SD_COVER_MAX`
  (network) to `0.1`; both are gaps with zero force.
- `sd` returned for `invalid`: NaN, with status `3`, so an unchecked status
  fails visibly. *Confirmed (user, first try).*
- Limitation: `too_deep` can only use the predicted `sd`; the exact value is
  not known at runtime. A misclassification near `-0.1` is never larger than
  the model's own error.

### R03.5 Training data [10]

- Training SHALL read a spec 02 dataset folder (`data.npz` and
  `provenance.json`) and SHALL verify the sha256 of `data.npz` against the
  provenance before use.
- The training target is the stored `h`; `sd`, `orientation_id`,
  `component`, `sd_part`, `delta` are used for splitting and metrics.

### R03.6 Train/validation split [4]

- Split by `orientation_id`: no orientation in both sets.
- Seeded and recorded; fractions configurable, default 0.8 / 0.2.

### R03.7 Loss

- The loss SHALL be defined on the error of `h` in canonical units. Its form
  and any weighting are design choices, recorded per run.

### R03.8 Metrics [5]

Each run SHALL report, on the validation set:

- `h` error: max, 99th percentile, median, RMS of `|h_pred - h|`; also per
  component and per decade of `Δ`.
- `sd` error per zone (`accuracy`, `gap_safety`) and per decade of `|sd|`
  for each sign.
- Sign safety (spec 01 R01.17): false-contact rate, missed-contact rate,
  largest spurious overlap, at the thresholds `DELTA_SIGN` in
  `{0, 1e-5, 1e-4, 1e-3}`.
- Continuity (spec 01 R01.18): largest `|sd(q) - sd(-q)|` (must be 0), and
  the largest step change of predicted vs exact `sd` along a fixed
  continuous tumbling path.
- Precision (S03.4): largest difference between the float32 network and the
  exported float64 model on the validation set.
- *[OPEN, user, after baseline]* Tolerances for these metrics (spec 01
  R01.15, R01.17, R01.18). No pass/fail threshold is assumed now.

### R03.9 Precision *(S03.4)* [7]

- Training MAY use float32.
- The exported model SHALL hold float64 weights and compute `u`, `h` and
  `position_z - h` in float64.

### R03.10 Reproducibility [12]

- All seeds SHALL be recorded: NumPy, PyTorch, split, initialization,
  batch order.
- Deterministic settings SHALL be used where the library offers them. CPU
  runs SHALL be bitwise reproducible; GPU runs SHALL reproduce the final
  validation metrics within a tolerance stated in the design.

### R03.11 Artifacts and provenance [8, 12]

- Each run SHALL write a new folder `interactions/ss_cube-wall/output/run_XXX/`
  and never overwrite (spec 04 layout).
- Contents: the TorchScript model (float64), the network `state_dict`,
  `scalers.dat` (R03.16, *[changed 2026-10-04, S03.10]*), and a
  machine-readable run record holding: dataset path and sha256, git commit
  and dirty flag, Python/NumPy/PyTorch versions, device, seeds, all
  hyperparameters, the loss curve, the metrics of R03.8, start and end time
  in UTC, spec revisions, and the sha256 of each artifact file.
- No joblib files (S03.5).

### R03.12 Cost [12]

- Without user approval, training runs are limited to smoke runs: at most
  `1e4` rows and at most 10 minutes of wall time.
- Larger runs need user approval (constitution 9).

### R03.13 Errors [12]

- IF the configuration is invalid, the dataset is missing, its checksum
  does not match, or the loss becomes non-finite, THEN training SHALL stop
  with a descriptive error and leave no run folder that looks complete.

### R03.14 Hyperparameters *(S03.3)* [6]

- Architecture, optimizer, learning rate, batch size, epochs and stopping
  rule SHALL be configurable and recorded per run.
- Starting values are given in the design, not here.

### R03.15 Reuse by the learning-curve study *(S03.6)*

- Training SHALL be callable for a given dataset and configuration and
  return the run folder and metrics, so spec 07 can run it on several
  dataset sizes.

### R03.16 `scalers.dat` for C++ *[new 2026-10-04, S03.10, user: option A]*

- Each run SHALL write `scalers.dat`, plain text, next to `model.pt`.
- Its values SHALL equal the scaling and constants stored inside the
  exported model (checked by a test).
- It SHALL be readable by the existing key/value C++ loader of the
  reference code: a key line followed by a value line; keys `y_mean`,
  `y_scale`, `eps` present; `y_scale != 0`; other keys allowed (ignored by
  that loader).
- It SHALL carry a format version and state what the scaling applies to,
  so it cannot be mistaken for the old log-transform file.
- Floating-point values SHALL be written with 17 significant digits (exact
  float64 round trip).
- The model output `sd` is final. C++ code SHALL NOT apply `scalers.dat`
  values to the model output a second time. (Constraint for spec 06.)
- Exact layout: design decision D03.1.

## Numerical and physical requirements

- Canonical units (`L = 1`); real units only through spec 01 R01.7.
- Exported computation in float64 (R03.9).
- Sign convention: spec 01 R01.9 (`deltan = sd`).

## Edge cases

- `qw = 0` and `q` vs `-q` (R03.3).
- `position_z` exactly at `Z_PREFILTER`: `in_range` (pre-filter uses `>`).
- Batch of one row; rows mixing statuses in one batch.
- Corner-down orientations: `h` at its maximum `sqrt(3)/2`.
- Near-kink orientations: `h` not differentiable; error expected to be
  largest there (reported per `Δ` decade).

## Out of scope

- Learning-curve study and dataset-size choice (spec 07).
- Full-scale training without approval.
- Evaluation on fixed held-out test sets (spec 04).
- C++ inference and the LIGGGHTS integration (spec 06, separate project).
- Contact point and normal (S03.7).

## Completion criteria

- A smoke run on `dataset_001` writes `run_001` with all R03.11 artifacts.
- The exported model passes interface tests: shapes and dtypes;
  bit-identical output for `q` and `-q`; status `far`, `too_deep`,
  `invalid` cases; float64 output; runs without Python-side files.
- The R03.8 metrics are reported, labelled as a smoke test, not a result.
- The user approves this spec.

## Open questions

- Tolerances for R03.8 metrics: after the baseline (spec 04 / spec 07).
- R03.4 conventions are a first try; revisit after the first LIGGGHTS trial
  (integration project).
- GPU reproducibility tolerance: set in the design.
