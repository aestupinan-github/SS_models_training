# Spec 02 — ss_cube-wall data generation

## Requirements

### R02.1 Reference solution

- Exact analytical cube support function: rotate the 8 vertices by the quaternion, find `min_world_z`, `overlap = max(0, -min_world_z)`.
- No geometric approximation.

### R02.2 Sampling strategy — hybrid

Three components:

1. **Log-uniform overlap targets** — pick target overlap values log-uniformly in `[1e-5, 0.5]`, invert the support function to find exact `position_z` for each target. Guarantees dense small-overlap coverage.
2. **Critical orientations** — flat, principal (45/90/135/180° about x/y/z), corner-like (54.7356° about body diagonals), edge-like. Capture support-function topology.
3. **Random orientations** — uniform random rotations, swept over z-trajectories.

### R02.3 Z-trajectory

- For each orientation, compute `z_touch = -min(rotated_vertex_z)`.
- Sample `position_z` from `z_touch` down to `z_touch - max_overlap`.
- Dense sampling near contact.

### R02.4 Overlap range

- `[1e-5, 0.5]` canonical units.

### R02.5 Output format

- CSV with columns: `position_z`, `Q_orientation.w()`, `Q_orientation.x()`, `Q_orientation.y()`, `Q_orientation.z()`, `overlap(m)`.

### R02.6 Seeds

- Fixed, recorded random seed for all stochastic steps (orientation sampling, trajectory offsets, shuffling).

### R02.7 Dataset size

- Configurable target. Start at 100k; may increase based on accuracy needs.

### R02.8 Provenance record

- Each dataset has a `provenance.json` sidecar recording: script version, git commit, seed, sampling strategy, ranges, sample count, generation timestamp.
