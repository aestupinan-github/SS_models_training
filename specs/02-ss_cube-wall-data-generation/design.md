# Spec 02 — Design

## Sampling approach

The key challenge is dense coverage of small penetrations. The hybrid strategy:

1. **Log-uniform penetration targets** — sample penetration magnitude `p_target` log-uniformly in `[1e-5, 0.5]`. For each target, invert the support function: find `position_z` such that `compute_signed_distance(q, position_z) = -p_target`. This guarantees uniform log-space coverage.

2. **Critical orientations** — fixed set of orientations that capture the support-function topology (flat, principal, corner, edge). Each gets a full z-trajectory.

3. **Random orientations** — uniform random rotations via scipy `R.random()`, each with a z-trajectory.

4. **Gap samples** — for each orientation, sample `position_z` uniformly in `(z_touch, z_touch + 0.1]` to give the model coverage of the no-contact side.

## Support-function inversion

Since `signed_distance = min(rotated_vertex_z) + position_z = -z_touch + position_z`, inversion is exact:

- `z_touch = -min(rotated_vertex_z)`
- `position_z = z_touch + signed_distance_target`

For a penetration target of magnitude `p_target`, use `signed_distance_target = -p_target`, giving `position_z = z_touch - p_target`.

## Provenance

Each dataset CSV has a companion `provenance.json`:

```json
{
  "script_version": "generate_cubes_v1.py",
  "git_commit": "<hash>",
  "seed": 42,
  "sampling_strategy": "hybrid_loguniform",
  "penetration_range": [-0.5, -1e-5],
  "gap_range": [0.0, 0.1],
  "n_samples": 100000,
  "generated": "2026-10-02T14:30:00"
}
```
