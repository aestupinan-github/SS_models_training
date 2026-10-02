# ss_cube-wall

Cube-vs-plane contact interaction surrogate. The cube is a **testbed** for the
surrogate method (not the production shape). The exact cube solution is the
permanent reference baseline.

## Status

Phase 1, spec 01 (interaction contract): revision 2 implemented and approved
(2026-10-03). Data generator (spec 02), training (spec 03) and evaluation (spec 04)
not started.

## Contents

- `python/cube_wall.py` — exact reference solution and contract helpers:
  `canonicalize_quaternion`, `z_touch` (g(q)), `compute_signed_distance`,
  `to_canonical`, `from_canonical`, `classify_sd`, `is_far`.
- `python/smoke_contract.py` — contract smoke test (software check against the
  exact solution, not a scientific result).
- `data/`, `output/`, `figures/` — empty for now.

## Contract summary (see spec 01)

- Quaternion `(qw, qx, qy, qz)`, scalar first, body to world, normalized,
  canonical `qw >= 0` (tie-break at `qw == 0`).
- Signed distance `sd = position_z - g(q)`: positive = gap, negative = overlap.
  LIGGGHTS `deltan = sd`.
- Canonical size `L = 1` (edge length). Rescale: `z/L_actual` in, `sd*L_actual` out.
- Covered range: `sd` in `[-0.1, 0.1]`. Outside: reported, not clamped.

## Commands (from the project root)

```
../env_folder/bin/python -B -m unittest discover tests
../env_folder/bin/python -B interactions/ss_cube-wall/python/smoke_contract.py
```

## Specs

`specs/01-ss_cube-wall-interaction-contract/`, `02-...-data-generation/`,
`03-...-training/`, `04-...-evaluation-and-artifacts/`.
