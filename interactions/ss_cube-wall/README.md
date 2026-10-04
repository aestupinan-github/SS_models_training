# ss_cube-wall

Cube-vs-plane contact interaction surrogate. The cube is a **testbed** for the
surrogate method (not the production shape). The exact cube solution is the
permanent reference baseline.

## Status

- Spec 01 (interaction contract): revision 3 implemented and approved
  (2026-10-03).
- Spec 02 (data generation): generator implemented (T02.1-T02.8); first smoke
  dataset written. Full-scale generation (T02.9) waits for the spec 03
  learning-curve study and user approval.
- Spec 03 (training) and spec 04 (evaluation): not started.

## Contents

- `python/cube_wall.py` — exact reference solution and contract helpers:
  `canonicalize_quaternion`, `z_touch` (g(q)), `compute_signed_distance`,
  `to_canonical`, `from_canonical`, `classify_sd`, `is_far`, `classify_zone`,
  `support_height` (shape-general, any point set), shape constants
  `R_CIRCUMSCRIBED`, `H_MIN`, `Z_PREFILTER`, `SD_COVER_MAX`.
- `python/smoke_contract.py` — contract smoke test (software check against the
  exact solution, not a scientific result).
- `python/generate_dataset.py` — dataset generator (spec 02): orientation
  mixture (`uniform`, `near_kink`, `special`), five `sd` parts, labels from the
  exact reference, coverage table, acceptance checks, versioned
  `data/dataset_XXX/` folders with `data.npz`, optional `data.csv` and
  `provenance.json`. Never overwrites.
- `data/` — generated datasets (ignored by git; listed below).
- `output/`, `figures/` — empty for now.

## Datasets

Generated datasets are not committed (`.gitignore`). Each is reproducible from
its seed and the git commit in its `provenance.json`.

| Folder | Rows | Orientations x K | Seed | Files | Purpose | Commit |
|---|---|---|---|---|---|---|
| `data/dataset_001` | 10000 | 1250 x 8 | 20261003 | `data.npz`, `data.csv`, `provenance.json` | Smoke dataset (T02.8): software check of the generator; small CSV for later C++ tests. Not a training dataset, not a scientific result. All 9 acceptance checks pass. | `811e60d` |

Regenerate: `../env_folder/bin/python -B interactions/ss_cube-wall/python/generate_dataset.py --n-orient 1250 --k 8 --seed 20261003 --csv`
(at commit `811e60d`; gives an identical `data.npz`, sha256 `01f2f692fcf9...`).

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
../env_folder/bin/python -B interactions/ss_cube-wall/python/generate_dataset.py --help
```

## Specs

`specs/01-ss_cube-wall-interaction-contract/`, `02-...-data-generation/`,
`03-...-training/`, `04-...-evaluation-and-artifacts/`.
