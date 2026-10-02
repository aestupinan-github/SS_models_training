# Spec 02 — Design

## Sampling approach

The key challenge is dense coverage of small overlaps. The hybrid strategy:

1. **Log-uniform overlap targets** — sample `overlap_target` log-uniformly in `[1e-5, 0.5]`. For each target, invert the support function: find `position_z` such that `compute_overlap(q, position_z) = overlap_target`. This guarantees uniform log-space coverage.

2. **Critical orientations** — fixed set of orientations that capture the support-function topology (flat, principal, corner, edge). Each gets a full z-trajectory.

3. **Random orientations** — uniform random rotations via scipy `R.random()`, each with a z-trajectory.

## Support-function inversion

For a given orientation and target overlap `o_target`:
- `z_touch = -min(rotated_vertex_z)`
- `position_z = z_touch - o_target`

This is exact because the support function is piecewise-linear in z.

## Provenance

Each dataset CSV has a companion `provenance.json`:

```json
{
  "script_version": "generate_cubes_v1.py",
  "git_commit": "<hash>",
  "seed": 42,
  "sampling_strategy": "hybrid_loguniform",
  "overlap_range": [1e-5, 0.5],
  "n_samples": 100000,
  "generated": "2026-10-02T14:30:00"
}
```
