# Spec 04 — ss_cube-wall evaluation and artifacts

## Requirements

### R04.1 Evaluation test cases

1. **In-domain** — random poses and signed distances within the training distribution.
2. **Off-anchor** — poses/signed distances not seen during training (held-out orientations, edge/corner cases).
3. **Slope check** — verify the surrogate's gradient `d(signed_distance)/d(z)` matches the analytical slope. Since `signed_distance = -z_touch + position_z`, the exact slope is `+1`.
4. **Scale rescaling** — verify nondimensionalization: train at L=1, query at different particle sizes, confirm the rescale-by-half-size rule.

### R04.2 Fixed test cases

- Each test case is a fixed, documented set of poses stored under `interactions/ss_cube-wall/data/test_cases/`.
- Generated once with a fixed seed; reproducible across runs.

### R04.3 Slope check method

- Exact analytical derivative of the support function (not finite difference).
- The signed distance is linear in z with slope `+1`.

### R04.4 Scale rescaling test

- Sweep over at least 3 particle sizes (e.g. 0.5x, 1x, 2x canonical).
- Verify: `signed_distance_actual = signed_distance_model * half_size`.

### R04.5 Artifact layout

Under `interactions/ss_cube-wall/output/`:

- `run_XXX/models/` — `.pt`, `.pth`, `.save`, `scalers.dat`
- `run_XXX/configs/` — training config, dataset provenance
- `run_XXX/logs/` — training logs, evaluation results
- `run_XXX/results/` — evaluation metrics, comparison tables

### R04.6 Naming convention

- Include interaction name, date, and config identifier (e.g. `ss_cube-wall_2026-10-02_v1.pt`).

### R04.7 Overwrite policy

- Never silently overwrite.
- New runs get new versioned folders (`run_001/`, `run_002/`).

### R04.8 Figures

- Under `interactions/ss_cube-wall/figures/`: pred-vs-true, error histograms, loss curves.

### R04.9 Python-only evaluation

- Spec 04 evaluation is Python-only (analytical vs surrogate).
- C++ cross-check deferred to spec 06.
