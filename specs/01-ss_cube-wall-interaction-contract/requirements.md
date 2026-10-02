# Spec 01 — ss_cube-wall interaction contract

## Status

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

### R01.11 Signed distance range *[changed]*

- Penetration range covered: `[-0.1, 0)` canonical units (up to 10% of the
  edge length). *Confirmed (user):* overlaps up to `1e-1` must be covered.
  Previous value was `-0.5`.
- Gap range covered: `(0, 0.1]` canonical units.
- **Near-contact band** `|sd| < 1e-5`: part of the covered range. Every
  contact passes through it, so spec 02 samples it explicitly (density
  defined there).

### R01.12 Position range *[changed]*

- For a given orientation, `position_z` ranges from `g(q) + 0.1` (gap) down
  to `g(q) - 0.1` (penetration).

### R01.13 Out-of-range queries *[new]*

- A query is classified by its exact `sd` (reference) as `far` if
  `sd > 0.1`, `too_deep` if `sd < -0.1`, else `in_range`.
- **Shape-general pre-filter:** if canonical `position_z > R_circ + 0.1`,
  where `R_circ = sqrt(3)/2` is the circumscribed-sphere radius, then
  `sd > 0.1` for every orientation, so the query is `far` (no contact)
  without running the model.
- A `too_deep` query is outside the trained range: it is reported, never
  silently clamped. The reporting mechanism for the surrogate is defined in
  spec 03/06.

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
- Design choices in later specs should rely only on structure shared with
  other convex shapes (the support function on the body-frame direction
  `u = R(q)^T e_z`), not on cube-specific closed forms.

## Numerical and physical requirements

- Reference computation uses `float64`.
- Dimensionless quantities (`L = 1`). Real units enter only through R01.7.

## Edge cases

- Face-down (`g = 0.5`), edge-down (`g = sqrt(2)/2`), corner-down
  (`g = sqrt(3)/2`) orientations.
- `qw == 0` (180-degree rotations): two quaternion signs for one orientation.
- Zero-norm or non-finite quaternion: rejected (R01.4).
- Non-unit quaternion: normalized (R01.4).

## Out of scope

- Contact point and contact normal (deferred).
- Any cube-cube interaction, other shapes, moving or rotating walls.
- LIGGGHTS integration and the size-parameter mapping.

## Completion criteria

- The reference implementation and tests cover R01.4, R01.5, R01.7, R01.13,
  R01.14.
- The contract smoke test passes and is reported as a smoke test, not as a
  scientific result.
- The user approves this spec.

## Open questions

- Is LIGGGHTS `Q_orientation` body-to-world? (integration project)
- Which LIGGGHTS size parameter maps to `L_actual`? (integration project)
- Numeric accuracy tolerance (R01.15), to be set by the user after the
  baseline.
