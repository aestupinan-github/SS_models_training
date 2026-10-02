# Spec 04 — Design

## Test case generation

Fixed pose sets generated once with a fixed seed and stored as CSV under `data/test_cases/`:

- `in_domain.csv` — random poses within training distribution.
- `off_anchor.csv` — held-out orientations, edge/corner cases.
- `scale_rescaling.csv` — poses for multiple particle sizes.

## Slope check

The support function `h(z) = min_world_z(z)` is piecewise-linear in z. When a single vertex is the support point, `dh/dz = -1`, so `d(overlap)/dz = +1`. The surrogate's gradient is computed via autograd and compared to the analytical value.

## Scale rescaling

For a particle with half-size `a`:
- Query input: `z_query = z_actual / a`
- Model output: `overlap_model`
- Rescaled: `overlap_actual = overlap_model * a`

Test at `a ∈ {0.25, 0.5, 1.0}` (i.e. 0.5x, 1x, 2x canonical).

## Artifact versioning

Each training/evaluation run creates `output/run_XXX/` with a monotonically increasing counter. Contents:

```
output/run_001/
  models/
  configs/
  logs/
  results/
```
