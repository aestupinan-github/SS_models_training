# ss_cube-wall

Cube-vs-plane contact interaction surrogate. The cube is a **testbed** for the
surrogate method (not the production shape). The exact cube solution is the
permanent reference baseline.

## Status

Phase 1, spec 01 (interaction contract): revision 3 implemented and approved
(2026-10-03). Data generator (spec 02), training (spec 03) and evaluation (spec 04)
not started.

## Contents

- `python/cube_wall.py` — exact reference solution and contract helpers:
  `canonicalize_quaternion`, `z_touch` (g(q)), `compute_signed_distance`,
  `to_canonical`, `from_canonical`, `classify_sd`, `is_far`, `classify_zone`,
  `support_height` (shape-general, any point set), shape constants
  `R_CIRCUMSCRIBED`, `H_MIN`, `Z_PREFILTER`, `SD_COVER_MAX`.
- `python/smoke_contract.py` — contract smoke test (software check against the
  exact solution, not a scientific result).
- `data/`, `output/`, `figures/` — empty for now.

## Contract summary (see spec 01)

- Quaternion `(qw, qx, qy, qz)`, scalar first, body to world, normalized,
  canonical `qw >= 0` (tie-break at `qw == 0`).
- Signed distance `sd = position_z - g(q)`: positive = gap, negative = overlap.
  LIGGGHTS `deltan = sd`.
- Canonical size `L = 1` (edge length). Rescale: `z/L_actual` in, `sd*L_actual` out.
- Pre-filter: canonical `z > Z_PREFILTER = sqrt(3)/2 + 0.1` means no contact.
- Accuracy zone `sd` in `[-0.1, 0.1]`; gap safety zone `(0.1, SD_COVER_MAX ≈ 0.466]`
  (sign must be right). Every query that reaches the model is covered.
  Deeper than `-0.1`: reported, not clamped.
- Surrogate must be sign-safe (R01.17) and continuous, identical for `q`
  and `-q` (R01.18).

## Commands (from the project root)

```
../env_folder/bin/python -B -m unittest discover tests
../env_folder/bin/python -B interactions/ss_cube-wall/python/smoke_contract.py
```

## Specs

`specs/01-ss_cube-wall-interaction-contract/`, `02-...-data-generation/`,
`03-...-training/`, `04-...-evaluation-and-artifacts/`.
