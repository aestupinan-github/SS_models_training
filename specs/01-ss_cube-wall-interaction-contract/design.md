# Spec 01 — Design

## Geometry

Cube of half-size 0.5 centered at `(0, 0, position_z)` in world frame. Wall is the plane `z = 0`.

## Quaternion convention

- Order: `(qw, qx, qy, qz)` — scalar first, matching LIGGGHTS.
- Canonicalization: if `qw < 0`, negate all four components. This resolves the `q` / `-q` duality.

## Overlap computation (reference solution)

1. Rotate the 8 body-frame vertices by the quaternion.
2. Find `min_world_z = min(rotated_vertex_z + position_z)`.
3. `overlap = max(0, -min_world_z)`.

This is the exact analytical cube support function — no geometric approximation.

## Scaling

- Train at canonical L = 1.
- At query time: `z_query = z_actual / half_size`, `overlap_actual = overlap_model * half_size`.

## Model I/O

| | |
|---|---|
| Input | `[position_z, qw, qx, qy, qz]` (5 features) |
| Output | `overlap` (1 scalar, canonical units) |
