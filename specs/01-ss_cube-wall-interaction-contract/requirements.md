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

### R01.5 Overlap definition

- `overlap = max(0, -min_world_z_of_vertices)`.
- Positive when any vertex penetrates the wall plane.
- Zero when no contact.

### R01.6 Model input

- `[position_z, qw, qx, qy, qz]` — 5 features, canonicalized quaternion.

### R01.7 Model output

- Single scalar: overlap in canonical units (L = 1).
- At query time: divide input `z` by particle half-size, multiply output overlap back.

### R01.8 Contact and no-contact behavior

- Only positive-overlap (contact) samples in training data.
- Zero/negative overlap = no contact, not part of the regression target.
- LIGGGHTS convention: `actual_overlap < 0`; surrogate overlap is positive.

### R01.9 Sign convention

- Surrogate overlap is positive.
- LIGGGHTS `deltan = -overlap_surrogate`.

### R01.10 Configuration-specific

- One fixed cube shape, one fixed plane. No generalization across shapes or walls.

### R01.11 Overlap range

- Training overlap range: `[1e-5, 0.5]` canonical units.

### R01.12 Position range

- `position_z` sampled from `z_touch` (just before contact, orientation-dependent) down to `z_touch - max_overlap`.
- `z_touch = -min(rotated_vertex_z)` for a given orientation.
