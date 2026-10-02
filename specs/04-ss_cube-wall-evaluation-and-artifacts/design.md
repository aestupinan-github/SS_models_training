# Spec 04 — Design

## Test case generation

Fixed pose sets generated once with a fixed seed and stored as CSV under `data/test_cases/`:

- `in_domain.csv` — random poses within training distribution.
- `off_anchor.csv` — held-out orientations, edge/corner cases.
- `scale_rescaling.csv` — poses for multiple particle sizes.

## Slope check

The signed distance is `s(z) = min(rotated_vertex_z) + z`, which is linear in `z` with slope `ds/dz = +1`. The surrogate's gradient is computed via autograd and compared to this exact analytical value.

## Scale rescaling

For a particle with half-size `a`:
- Query input: `z_query = z_actual / a`
- Model output: `signed_distance_model`
- Rescaled: `signed_distance_actual = signed_distance_model * a`

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
