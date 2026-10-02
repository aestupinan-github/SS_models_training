# Spec 01 — ss_cube-wall interaction contract

## Status

**Revision 3, 2026-10-03: amendment APPROVED by the user** ("ok" to: R01.16
convex hull generalisation; no unseen inputs (R01.11, R01.12, R01.13); sign
safety (R01.17); continuity (R01.18)). Changes of revision 3 are marked
*[rev 3]*.

**Revision 2, 2026-10-03. APPROVED by the user (2026-10-03, "ok" after
review of the implemented revision).** Earlier acceptance to proceed (user replies:
L is the characteristic length; near-contact and out-of-range proposals,
baseline-first accuracy tolerance, and code-after-approval all "ok"; then
"proceed with finishing 01"). Supersedes the first version (git history,
commit `af184cc`). Changes are marked *[changed]* or *[new]*. Items still
marked *[OPEN]* belong to other projects or to a later measurement.

## Context and objective

*Confirmed (user).* The end goal is a new LIGGGHTS particle-shape type whose
overlap is computed by a trained surrogate, evaluated from C++ at runtime.
Accuracy of the overlap comes first. `ss_cube-wall` is a **testbed** for the
surrogate method, not the production shape. Contact point and normal are a
later feature.

This spec defines the interface contract and the exact reference solution of
the `ss_cube-wall` interaction. It does not define sampling (spec 02),
training (spec 03), or evaluation (spec 04).

## Definitions

- **Pose:** cube centre height `position_z` and orientation quaternion `q`.
- **Body frame:** frame fixed to the cube, axes along the cube edges.
- **World frame:** `z` is the wall normal, pointing away from the wall.
- **`L`:** characteristic length of the particle. For the cube it is the edge
  length. Canonical `L = 1`. *Confirmed (user).*
- **`g(q)`** (`z_touch`): height of the cube centre at which the cube just
  touches the wall, for orientation `q`. Depends on orientation only.
- **Signed distance `sd`:** see R01.5.

## Requirements

### R01.1 Cube geometry

- Cube with canonical edge length `L = 1` (half-size 0.5). *[changed]* The
  characteristic size used for rescaling is the **edge length** `L`.
- Vertices at `(±0.5, ±0.5, ±0.5)` in the body frame.

### R01.2 Wall geometry

- Infinite plane at `z = 0`, normal pointing in `+z`.
- The cube is on the `z > 0` side.

### R01.3 Coordinate frame

- World frame: `z` is the wall normal.
- `position_z` is the height of the cube centre above the wall plane.

### R01.4 Pose representation *[changed]*

- Orientation is a quaternion `(qw, qx, qy, qz)` (scalar first).
- **Rotation sense:** the quaternion rotates body-frame vectors into the
  world frame (active rotation, body to world).
- **Normalization:** input quaternions are normalized to unit length before
  use.
- **Canonicalization** (resolves the `q` / `-q` duality): after
  normalization, if `qw < 0`, negate all four components. If `qw == 0`, negate
  all four components so that the first nonzero component among
  `(qx, qy, qz)` is positive. (`qw == 0` is tested exactly on the normalized
  input.)
- **Invalid input:** a quaternion with zero norm or any non-finite component
  is rejected with a descriptive error. It must not produce NaN silently.
- *[OPEN, integration project]* Whether LIGGGHTS `Q_orientation` is
  body-to-world must be checked there. This spec defines the contract in
  terms of the sense above and does not depend on the answer.

### R01.5 Signed distance definition *[changed]*

- `sd` is the signed distance between the cube and the wall plane, defined
  through the cube's support function: the height above the plane of the
  cube point nearest to it.
- For the cube (a polyhedron) and a plane this is
  `sd = position_z + min over vertices of (world z offset of the vertex)`.
  Equivalently `sd = position_z - g(q)` with
  `g(q) = -min_v (R(q) v)_z`.
- Positive = gap (no contact), negative = penetration, zero = touching.
- `sd` is exact: no geometric approximation.

### R01.6 Model input

- `[position_z, qw, qx, qy, qz]`: 5 features, canonicalized quaternion
  (R01.4), `position_z` in canonical units (R01.7).

### R01.7 Model output and rescaling *[changed]*

- Single scalar: signed distance in canonical units (`L = 1`).
- **Rescaling:** for a cube of edge length `L_actual`, query the model with
  `z_canonical = position_z / L_actual` and obtain
  `sd_actual = sd_canonical * L_actual`.
- The mapping from the simulation code's size parameter (for example a
  half-size) to `L_actual` belongs to the integration project. *[OPEN,
  integration project]* The user believes the LIGGGHTS parameter is the
  half-size, in which case `L_actual = 2 * half_size`. This spec does not
  depend on the answer.
- The model output is **not clamped** (previous `[-0.5, +0.1]` clamp
  removed). Range handling is defined in R01.13.

### R01.8 Contact and no-contact behavior

- Training data includes both penetration (negative) and gap (positive) samples.
- Penetration is the physically meaningful quantity for DEM.
- Gap samples teach the model the sign boundary. The overlap depends on
  position and orientation together: a flat cube above the wall can be in
  gap while a tilted cube at the same height overlaps.

### R01.9 Sign convention

- Positive = gap, negative = penetration.
- *Confirmed (user).* LIGGGHTS `deltan` is negative for overlap, so
  `deltan = sd`.

### R01.10 Configuration-specific

- One fixed cube shape, one fixed plane. No generalization across shapes or walls.

### R01.11 Signed distance range *[changed, rev 3]*

- Penetration range covered: `[-0.1, 0)` canonical units (up to 10% of the
  edge length). *Confirmed (user):* overlaps up to `1e-1` must be covered.
  Previous value was `-0.5`.
- **Accuracy zone:** `-0.1 <= sd <= 0.1`. The accuracy requirement R01.15
  applies here.
- **Gap safety zone** *[rev 3]*: `0.1 < sd <= SD_COVER_MAX`, where
  `SD_COVER_MAX = Z_PREFILTER - H_MIN` (R01.13) is the largest `sd` of any
  query that is not pre-filtered. For the cube
  `SD_COVER_MAX = sqrt(3)/2 + 0.1 - 0.5 ≈ 0.4660`. The sign-safety
  requirement R01.17 applies here; R01.15 does not.
- **Near-contact band** `|sd| < 1e-5`: part of the covered range. Every
  contact passes through it, so spec 02 samples it explicitly (density
  defined there).
- **No unseen inputs** *[rev 3]*: every query that reaches the model (not
  pre-filtered, R01.13) lies in the covered range
  `[-0.1, SD_COVER_MAX]`, except `too_deep` queries (R01.13). Training data
  must cover this whole range.

### R01.12 Position range *[changed, rev 3]*

- For a given orientation `q`, the covered `position_z` range is
  `g(q) - 0.1 <= position_z <= Z_PREFILTER` (R01.13). In terms of `sd`:
  `-0.1 <= sd <= Z_PREFILTER - g(q)`, which is at most `SD_COVER_MAX`.

### R01.13 Query zones and pre-filter *[new, changed rev 3]*

- **Shape constants** (canonical units):
  - `R_CIRC`: circumscribed-sphere radius about the centre
    (cube: `sqrt(3)/2`).
  - `H_MIN`: smallest value of `g(q)` over all orientations (cube: `0.5`,
    face-down).
  - `Z_PREFILTER = R_CIRC + 0.1`.
  - `SD_COVER_MAX = Z_PREFILTER - H_MIN` (cube: `≈ 0.4660`).
- **Pre-filter** (uses only runtime inputs): if canonical
  `position_z > Z_PREFILTER`, the query is "no contact" and the model is not
  evaluated. This is safe: then `sd > Z_PREFILTER - g(q) >= 0.1` for every
  orientation.
- **Zones** of a query, by its exact `sd` (used for training-data coverage and
  for evaluation):
  - `too_deep`: `sd < -0.1`. Outside the trained range. Reported, never
    silently clamped. The surrogate's reporting mechanism is defined in spec
    03/06.
  - `accuracy`: `-0.1 <= sd <= 0.1` (R01.15).
  - `gap_safety`: `0.1 < sd <= SD_COVER_MAX` (R01.17).
  - `beyond`: `sd > SD_COVER_MAX`. Only reachable above `Z_PREFILTER`, so
    always pre-filtered.

### R01.14 Reference solution and invariances *[new]*

The exact reference solution (R01.5) must satisfy the following. The
surrogate is tested against them within a tolerance given by R01.15.

- **Slope:** `d sd / d position_z = +1` exactly.
- **Yaw invariance:** `sd` does not change under rotation of the cube about
  the world `z` axis.
- **Cube symmetry:** `sd` does not change under the 24 proper rotations that
  map the cube onto itself (applied in the body frame).
- **Quaternion sign:** `q` and `-q` give the same `sd` and the same
  canonical quaternion.
- **Range of `g`:** `0.5 <= g(q) <= sqrt(3)/2` (face-down to corner-down).

### R01.15 Accuracy *[new]*

- Accuracy of `sd` is the first priority (constitution, principle 12).
- The tolerance for the surrogate is given as an absolute error on `sd` in
  canonical units, and separately as a relative error for `|sd| > 1e-3`.
- *[OPEN, user, after baseline]* The numeric tolerance is not fixed. A
  baseline error distribution is measured first (spec 04), then the user sets
  the tolerance from it. No tolerance is assumed until then.

### R01.16 Testbed status *[new]*

- The cube is a testbed for the surrogate method. The exact cube solution is
  the permanent reference baseline; every surrogate error is reported
  against it.
- *[rev 3]* *Confirmed (user):* the target shapes are arbitrary, given as an
  STL surface or a point set, and may be non-convex.
- *[rev 3]* For any rigid shape against a plane, `sd = position_z - h(q)`,
  where `h(q) = max_v (v . u)` is the support function of the shape's
  **convex hull** in the body-frame direction pointing towards the wall,
  `u = -R(q)^T e_z` (the maximum is over hull vertices `v`; the lowest point
  of any shape is a hull vertex). This holds for non-convex shapes too. For
  the cube, `h = g`.
- Design choices in later specs should rely only on this shape-general
  structure (point set or hull, support function, `R_CIRC`, `H_MIN`), not on
  cube-specific closed forms.

### R01.17 Sign safety *[new, rev 3]*

- Applies to the surrogate, for all queries that reach the model.
- A query with exact `sd > DELTA_SIGN` must not be predicted as contact
  (`sd_pred < 0`), and a query with exact `sd < -DELTA_SIGN` must not be
  predicted as gap.
- Reported metrics: false-contact rate, missed-contact rate, and the largest
  spurious overlap (largest `-sd_pred` over queries with exact
  `sd > DELTA_SIGN`).
- *[OPEN, user, after baseline]* `DELTA_SIGN` and the acceptable rates are not
  fixed. They are set by the user after the baseline (spec 04).

### R01.18 Continuity *[new, rev 3]*

- Applies to the surrogate.
- **Sign invariance:** the prediction for `q` and `-q` is identical (within
  floating-point round-off), whatever the canonicalization. Reason
  (*Confirmed*, measured): on a continuous tumbling path of 20000 steps of
  about 1 mrad, the canonical quaternion of R01.4 jumped 3 times, so a
  prediction that depends on the quaternion sign jumps along a continuous
  motion.
- **Path continuity:** along a continuous pose path sampled at small steps,
  the change of the prediction between consecutive steps is bounded by the
  change of the exact `sd` plus a tolerance.
- *[OPEN, user, after baseline]* The path-continuity tolerance is set after
  the baseline (spec 04). How the model meets R01.18 is a spec 03 design
  choice.

## Numerical and physical requirements

- Reference computation uses `float64`.
- Dimensionless quantities (`L = 1`). Real units enter only through R01.7.

## Edge cases

- Face-down (`g = 0.5`), edge-down (`g = sqrt(2)/2`), corner-down
  (`g = sqrt(3)/2`) orientations.
- `qw == 0` (180-degree rotations): two quaternion signs for one orientation.
- Zero-norm or non-finite quaternion: rejected (R01.4).
- Non-unit quaternion: normalized (R01.4).
- *[rev 3]* Queries just below `Z_PREFILTER`: largest reachable gap
  (`SD_COVER_MAX` for face-down).
- *[rev 3]* Continuous rotation through `qw = 0`: the canonical quaternion
  changes sign (R01.18).

## Out of scope

- Contact point and contact normal (deferred).
- Any cube-cube interaction, other shapes, moving or rotating walls.
- LIGGGHTS integration and the size-parameter mapping.

## Completion criteria

- The reference implementation and tests cover R01.4, R01.5, R01.7, R01.13,
  R01.14.
- *[rev 3]* R01.17 and R01.18 are surrogate requirements; their metrics are
  implemented in spec 04.
- The contract smoke test passes and is reported as a smoke test, not as a
  scientific result.
- The user approves this spec.

## Open questions

- Is LIGGGHTS `Q_orientation` body-to-world? (integration project)
- Which LIGGGHTS size parameter maps to `L_actual`? (integration project)
- Numeric accuracy tolerance (R01.15), to be set by the user after the
  baseline.
- `DELTA_SIGN`, acceptable sign-error rates (R01.17) and path-continuity
  tolerance (R01.18), to be set by the user after the baseline.
