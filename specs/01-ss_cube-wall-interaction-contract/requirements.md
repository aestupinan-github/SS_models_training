# Spec 01 — ss_cube-wall interaction contract

## Requirements

### R01.1 Cube geometry

- Axis-aligned cube, canonical characteristic length L = 1 (half-size 0.5).
- Vertices at `(±0.5, ±0.5, ±0.5)` in body frame.

### R01.2 Wall geometry

- Infinite plane at `z = 0`, normal pointing in `+z`.
- The cube approaches from `z > 0`.

### R01.3 Coordinate frame

- World frame: `z` is the wall normal.
- `position_z` is the cube center height above the wall plane.

### R01.4 Pose representation

- Cube orientation as quaternion `(qw, qx, qy, qz)` (wxyz order, matching LIGGGHTS `Q_orientation`).
- Canonicalization: enforce `qw >= 0` (flip all signs if `qw < 0`).

### R01.5 Signed distance definition

- `signed_distance = min_world_z_of_vertices`.
- Positive = gap (no contact), negative = penetration.
- Zero when the lowest vertex touches the wall plane.

### R01.6 Model input

- `[position_z, qw, qx, qy, qz]` — 5 features, canonicalized quaternion.

### R01.7 Model output

- Single scalar: signed distance in canonical units (L = 1).
- At query time: divide input `z` by particle half-size, multiply output signed distance back.
- Output clamped to `[-0.5, +0.1]`.

### R01.8 Contact and no-contact behavior

- Training data includes both penetration (negative) and gap (positive) samples.
- Penetration is the physically meaningful quantity for DEM.
- Gap samples teach the model the sign boundary.

### R01.9 Sign convention

- Model predicts signed distance: positive = gap, negative = penetration.
- LIGGGHTS `deltan = signed_distance` (negative in contact).

### R01.10 Configuration-specific

- One fixed cube shape, one fixed plane. No generalization across shapes or walls.

### R01.11 Signed distance range

- Penetration range: `[-0.5, -1e-5]` canonical units.
- Gap range: `(0, 0.1]` canonical units (20% of half-size).

### R01.12 Position range

- `position_z` sampled from `z_touch + 0.1` (gap) down to `z_touch - 0.5` (penetration).
- `z_touch = -min(rotated_vertex_z)` for a given orientation.
