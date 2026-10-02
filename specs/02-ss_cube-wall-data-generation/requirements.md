# Spec 02 — ss_cube-wall data generation

## Status

**Revision 2, 2026-10-03. APPROVED by the user (2026-10-03)**, including the
default fractions (R02.2, R02.3) and the `dataset_XXX` folder name (R02.8).
Supersedes revision 1 (git history). Addresses the 41 review findings of 2026-10-03 (finding numbers in
brackets). `design.md` and `tasks.md` are not updated until this file is
approved. Items marked *[OPEN]* need a user decision; items marked
*Proposal* are defaults the user may change.

## Context and objective

*Confirmed (user).* Training data must be as dense as possible, with overlap
accuracy first. The cube is a testbed; target shapes are arbitrary (STL or
point set, possibly non-convex). A fully trained model is the primary
approach.

This spec defines the **training datasets** for `ss_cube-wall`: what they
contain, how densely they cover the pose space, how they are stored, and how
each dataset is checked. The contract (frames, quaternion rules, `sd`,
zones) is defined by spec 01 revision 3 and is not repeated here [19].

## Scope

- In scope: training and validation-pool datasets for `ss_cube-wall`; small
  CSV exports of the same content for later C++ tests.
- Out of scope: fixed evaluation test sets (spec 04), training and the
  learning-curve study (spec 03), C++ inference (spec 06).

## Definitions

- **Row:** one pose `(position_z, q)` with its labels. Dataset size is
  counted in rows [25].
- **Orientation:** one canonical quaternion. A dataset has `N_orient`
  orientations and `K` rows per orientation, so `N_rows = N_orient × K`.
- **Support height `h(q)`:** spec 01 R01.16. `sd = position_z - h(q)`.
- **Support gap `Δ(q)`:** difference between the second-smallest and the
  smallest world height of the hull vertices for orientation `q`. `Δ = 0`
  where `h` is not differentiable (the lowest feature is an edge or a face).
  Shape-general [9, 38].
- **Component:** a named orientation-sampling rule (R02.3). Each row records
  its component [3].
- **Smoke dataset:** a dataset of at most `1e4` rows [28].

## Requirements

### R02.1 Labels from the exact reference

- THE GENERATOR SHALL compute `h(q)` and `sd` with the spec 01 reference
  solution, from a body-frame point set (cube: its 8 vertices). No
  approximation.
- THE GENERATOR SHALL store `sd` recomputed from the stored `position_z` and
  stored quaternion, so every stored row is self-consistent [17, 18].

### R02.2 Covered `sd` range and distribution

- Rows SHALL cover the spec 01 covered range `[-0.1, SD_COVER_MAX]` [1]:
  - **Accuracy zone, penetration:** `sd ∈ [-0.1, -1e-5]`, log-uniform in
    `|sd|`.
  - **Accuracy zone, gap:** `sd ∈ [1e-5, 0.1]`, log-uniform in `sd`.
  - **Near-contact band:** `|sd| < 1e-5`, both signs, uniform in `sd` [2].
  - **Exact contact:** `sd = 0` [14].
  - **Gap safety zone:** `sd ∈ (0.1, Z_PREFILTER - h(q)]`, uniform in `sd`.
    The interval depends on the orientation and is empty at corner-down
    (`Z_PREFILTER - h = 0.1`, measured minimum `0.100009` over 20000 random
    orientations). WHEN the interval is empty, the gap safety rows of that
    orientation SHALL be drawn from the accuracy-zone gap part instead.
- Range boundaries `-0.1` and `0.1` are inside the accuracy zone [15].
- No row SHALL have `sd < -0.1` or be above the pre-filter
  (`position_z > Z_PREFILTER`).
- *Proposal* (defaults): fractions of rows per `sd` part: penetration 0.40,
  gap 0.30, near-contact 0.05, exact contact 0.01, gap safety 0.24 [27].
  *Confirmed (user):* lower bound `1e-5` is fine for now.

### R02.3 Orientation sampling

- Orientations SHALL be drawn from a mixture of components, with
  configurable fractions [3, 5, 27]:
  - **`uniform`:** uniform on SO(3) (Haar measure) [13].
  - **`near_kink`:** orientations at small support gap `Δ`, log-uniform in
    `Δ` over `[1e-6, 1e-1]`, near both faces and edges [12]. Reason
    (*Confirmed*, measured): 1e6 uniform orientations give only 18 rows with
    `Δ ∈ [1e-6, 1e-5)` and 278 in `[1e-5, 1e-4)`.
  - **`special`:** the exact face-down, edge-down and corner-down
    orientations [9, 10, 11].
- Each orientation from every component SHALL then be rotated by a uniform
  random yaw about world `z` (does not change `sd`, R01.14), so the model
  sees all equivalent quaternions [11].
- *Proposal* (defaults): `uniform` 0.70, `near_kink` 0.29, `special` 0.01.
- The set of special orientations SHALL be defined through the shape (lowest
  feature is a face, an edge, or a vertex), not through cube-specific angles
  [10, 38].
- The generator SHALL accept a list of orientations to **exclude** (held out
  for evaluation, spec 04), and SHALL guarantee no row has an excluded
  orientation within a configurable angular distance [36].

### R02.4 Rows per orientation

- For each orientation, the `K` rows SHALL draw their `sd` values according
  to R02.2, then set `position_z = h(q) + sd` [4, 6].
- "Z-trajectory" in revision 1 is replaced by this rule [6].

### R02.5 Measurable density

- Each dataset SHALL report a **coverage table** [7, 8]:
  - rows per decade of `|sd|` for each sign, over `[1e-5, 1e-1]`;
  - rows per decade of `Δ` over `[1e-6, 1e-1]`;
  - rows per component and per `sd` part.
- Acceptance: every decade cell above SHALL contain at least `M_min` rows.
- *[OPEN]* `M_min` for full-scale datasets is set by the user after the
  learning-curve study (spec 03). For smoke datasets, `M_min = 1`.

### R02.6 Stored content and precision

- Columns, in this order, plain names [Confirmed (user)]:
  `position_z, qw, qx, qy, qz, sd, h, orientation_id, component` [20, 24].
  - `h` is stored so either learning target (spec 03) can use the data
    without regeneration [24].
  - `orientation_id`: integer, the same for all `K` rows of one
    orientation; needed for an orientation-based split (spec 03).
  - `component`: one of `uniform`, `near_kink`, `special`.
- All values are dimensionless canonical units (`L = 1`); no unit suffix
  [20, 21]. Frame, quaternion rules and sign: spec 01.
- Quaternions SHALL be stored normalized and canonical per R01.4 [16].
- Floating-point values SHALL be stored as float64 [22].

### R02.7 File formats

- **Training format:** NumPy `.npz`, one array per column [Confirmed (user)].
- **Test export:** the same generator SHALL be able to write a dataset as CSV
  with a header row, comma delimiter, the R02.6 columns, and floats with 17
  significant digits (exact float64 round trip), for C++ tests (spec 04/06)
  [Confirmed (user)] [23].
- CSV export SHALL be limited to smoke-size datasets unless the user
  approves more.
- Rows SHALL NOT be shuffled at generation time; the order is deterministic.
  Shuffling belongs to training (spec 03) [33].

### R02.8 Storage, versioning and overwrite

- Each dataset SHALL be written to its own new folder under
  `interactions/ss_cube-wall/data/` [23].
- THE GENERATOR SHALL refuse to write into an existing dataset folder; it
  never overwrites [29].
- *Proposal:* folder name `dataset_XXX` with an increasing counter.

### R02.9 Reproducibility

- One master seed per dataset. All random streams are derived from it in a
  documented, recorded way [32, 33].
- WHEN the same configuration, seed, and library versions are used, THE
  GENERATOR SHALL produce identical data arrays (bitwise) [32].
- Reproducibility across library versions or platforms is not required;
  the versions are recorded (R02.10).

### R02.10 Provenance

Each dataset SHALL have a machine-readable provenance record next to the
data, recording [34]:

- git commit and whether the working tree had uncommitted changes;
- Python, NumPy and SciPy versions;
- the full generation configuration (all counts, fractions, ranges,
  components, exclusion list or its checksum, master seed and derived
  seeds);
- the spec revision (`spec 01 rev 3`, `spec 02 rev 2`);
- generation timestamp in UTC (ISO 8601);
- a checksum of each data file;
- the coverage table (R02.5) and acceptance results (R02.12).

### R02.11 Invalid input and failures

- IF the configuration is invalid, THEN THE GENERATOR SHALL stop before
  writing anything and report which parameter is wrong [30]. Invalid
  includes: non-positive counts; fractions that are negative or do not sum
  to 1; ranges outside spec 01; lower bound not below upper bound; missing
  seed.
- IF a non-finite value appears, or the reference solution raises an error,
  or a write fails, THEN THE GENERATOR SHALL stop, report the cause, and
  leave no dataset folder that looks complete [31].
- IF the requested size exceeds the smoke limit and full-scale generation
  has not been approved, THEN THE GENERATOR SHALL refuse unless an explicit
  override is given [28].

### R02.12 Acceptance checks per dataset

Each generated dataset SHALL be checked, and the result recorded in the
provenance [35]:

- `|sd - (position_z - h_ref(q))| <= 1e-12` for every row, with `h_ref`
  recomputed by the reference solution;
- all rows inside the R02.2 ranges; none above `Z_PREFILTER`;
- no NaN or infinite values;
- quaternions unit-norm within `1e-12` and canonical (R01.4);
- coverage table meets `M_min` (R02.5);
- no row within the exclusion distance of an excluded orientation (R02.3).

These checks verify the data software. They are not a scientific
validation of any model.

### R02.13 Size and cost

- Size is configurable through `N_orient` and `K` [25].
- Smoke datasets (at most `1e4` rows) may be generated after this spec is
  approved. Larger datasets need user approval (constitution, principle 9)
  [28].
- The full-scale size is decided by the learning-curve study in spec 03
  [Confirmed (user)] [26].

### R02.14 Documentation and diagnostics

- Each generated dataset SHALL be listed in the interaction `README.md` with
  its folder, size, seed and purpose [37].
- The generator SHALL print a summary: size, components, coverage table,
  acceptance results, output path [40].

## Numerical and physical requirements

- float64 throughout generation and storage.
- Dimensionless (`L = 1`). Real units enter only through spec 01 R01.7.
- Label tolerance: `1e-12` (R02.12).

## Non-functional requirements

- No run-time or memory target now [40]. *Assumption:* generation is
  cheap (measured: 1e6 orientations with exact `h` in about 0.3 s); storage
  and training dominate.

## Edge cases

- `qw = 0` orientations (from `special` or yaw): stored canonical (R01.4).
- `special` orientations have `Δ = 0` exactly.
- Exact contact rows (`sd = 0`).
- Gap safety rows near `SD_COVER_MAX` (face-down at `Z_PREFILTER`).
- Corner-down: empty gap safety interval (R02.2).
- Orientations excluded for evaluation.
- Empty or all-excluded component: configuration error (R02.11).

## Out of scope

- Fixed evaluation test sets (spec 04).
- Shuffling, splitting, normalization (spec 03).
- Contact point and normal labels (deferred).
- Other shapes (copy the generator per R00.5 when a new interaction starts).

## Completion criteria

- A smoke dataset (at most `1e4` rows) is generated in `.npz`, and its CSV
  export, passing all R02.12 checks, with provenance (R02.10).
- Regenerating with the same seed gives identical arrays (R02.9).
- Invalid configurations are rejected without output (R02.11).
- The user approves this spec.

## Open questions

- `M_min` for full-scale datasets (R02.5): after the learning-curve study.
- Default fractions (R02.2, R02.3) are proposals; confirm or change.
- Angular exclusion distance for held-out orientations (R02.3): to be set
  with spec 04.
