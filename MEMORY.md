# MEMORY.md

## Confirmed facts

- Project lives in `Programs/SS_models_training/`.
- Git repository pushed to `git@github.com:aestupinan-github/SS_models_training.git` (branch `master`).
- **End goal (Confirmed, user):** a new LIGGGHTS particle-shape type whose
  overlap is computed by a trained surrogate, evaluated from C++ at runtime
  (no Python). The LIGGGHTS integration is a separate project, out of scope
  here. This repo delivers trained artifacts and their interface contract.
- **Priority (Confirmed, user):** accuracy of the overlap (signed distance)
  comes first. Contact point and contact normal are a later feature.
- LIGGGHTS `deltan` is negative for actual overlap (Confirmed, user), so
  `deltan = signed_distance`. Positive = gap, negative = penetration.
- LIGGGHTS integration with LibTorch already works (Confirmed, user), so
  TorchScript artifacts are a valid delivery format.
- Training data should be as dense as possible (Confirmed, user). Overlaps
  up to 1e-1 must be covered (Confirmed, user; units/fraction of L to be
  confirmed). Runtime matters but no optimisation now; keep LIGGGHTS
  compatibility in mind (Confirmed, user).
- The overlap depends on both position and orientation: a lifted cube that
  is flat has a gap but overlaps once tilted (Confirmed, user; checked
  numerically at z = 0.7).
- **The cube is a testbed (Confirmed, user)** for the surrogate method,
  not the production target. The exact cube solution stays as the permanent
  reference baseline; all errors are reported against it.
- **Target shapes (Confirmed, user):** arbitrary shapes given as an STL
  surface or a point set, possibly non-convex. Design must be shape-general.
- **Approach (Confirmed, user):** a fully trained model is the primary
  approach (more value for the paper). Embedding the shape (vertices) in the
  model is a fallback if the required accuracy is not reached.
- For any rigid shape against a plane: `sd = position_z - h(q)`,
  `h(q) = max_v v . u`, `u = -R(q)^T e_z`, over the convex-hull vertices
  (checked numerically; equals `z_touch` for the cube). Non-convexity does
  not change `sd` for wall interactions.
- Overlap range: up to 1e-1 = 10% of the edge length L (Confirmed, user).
  Lower bound of the near-contact band: `|sd| = 1e-5` for now (Confirmed,
  user); finer accuracy is a later concern with contact point/normal.
- Runtime cost depends on network size, not on training-data size
  (measured: about 50-100 us per single TorchScript query on CPU for 9k-265k
  parameters).
- Model input: `[position_z, qw, qx, qy, qz]` (5 features).
- Model output: signed distance only (single scalar, no contact head).
- Each model is configuration-specific: one fixed shape, one fixed plane.
- Train at canonical size L = 1 (edge length 1, half-size 0.5). At query
  time: `z / L_actual` in, `sd * L_actual` out (spec 01).
- `inspired_codes/` holds prior reference scripts. Inspiration only, partly
  unfinished, not imported at runtime, to be reviewed later.
- Code that exists: `interactions/ss_cube-wall/python/cube_wall.py`
  (reference solution: `canonicalize_quaternion`, `z_touch`,
  `compute_signed_distance`), `tests/test_overlap_contract.py`,
  `tests/test_structure.py`, `check_structure.py`.

## Unverified prior context

- Earlier generators used anchor heights with jitter and decade-stratified
  or continuous sampling. Model trained on log10 overlap target with EPS
  offset and scaler. Exported as TorchScript with ASCII scaler file.
- Earlier validation used off-anchor slope check and analytical vs surrogate
  vs C++ comparisons.
- These are design options to review, not fixed decisions.

## Current scope and status

- Phase 0: scaffold, constitution, specs 00-06 written.
- Phase 1: **spec 01 revision 3 approved and implemented** (2026-10-03).
  54 unit tests and the 17-check contract smoke test pass. Rev 3 added: zones
  and shape constants (`Z_PREFILTER`, `SD_COVER_MAX ≈ 0.466`, no unseen
  inputs below the pre-filter), shape-general `support_height`, sign safety
  (R01.17) and continuity (R01.18) as surrogate requirements.
- Spec 02 (data generation): reviewed (41 findings), requirements being
  redrafted. Spec 03, 04, 06: still revision 1, to be revised.
- Pending: spec 03 (replace transform, Option B, orientation-based split,
  learning-curve study, sign-invariant input, mini-batch training), spec 04
  (off-anchor definition, invariance tests, tumbling-path test for
  continuity and sign safety), spec 06, `.gitignore`/`.gitkeep`.

## Decisions and rationale

- Single-output model (signed distance only): simplifies the contract.
  Contact point/normal are deferred (user priority: overlap accuracy).
- Sampling (spec 02, revision 1, under review): penetration log-uniform;
  gap uniform in `(0, 0.1]`. Range is now `[-0.1, 0.1]`.
- No output clamp (spec 01 rev 2): out-of-range queries (`|sd| > 0.1`) are
  classified, not clamped. Pre-filter: canonical `z > sqrt(3)/2 + 0.1` is
  `far`.
- `L` is the characteristic length (edge length for the cube), Confirmed
  (user). Rescale: `z/L_actual`, `sd*L_actual`.
- Fresh C++ development, independent of `inspired_codes/`.

## Items under revision (found by spec review, 2026-10-03)

(Rescale rule and quaternion convention: resolved in spec 01 rev 2.)

- **Target transform.** `t(x) = sign(x) log10(|x| + EPS)` is not invertible
  for `|x| < 1` and jumps from about +15 to about -15 across contact. Spec 03
  (and spec 06 step 9, the `eps` scaler field) must be replaced with an
  invertible, monotone transform.
- **`.gitignore` and `.gitkeep`** do not cover `output/run_XXX/...`,
  `data/test_cases/...`, or `scalers.dat`.
- **Observed facts (checked numerically):** `signed_distance = z - z_touch(q)`,
  `z_touch = 0.5 * ||R^T e_z||_1`. It depends only on orientation, is
  invariant to yaw about z and to the 24 cube symmetries.

## Open questions

- **Size convention.** What LIGGGHTS's particle size parameter means for a
  cube (edge length or half-size). User believes half-size; to be
  investigated later. Contract must be independent of the answer: the
  rescale uses the edge length, and the integration project maps its size
  parameter to the edge length.
- Learning target: user leans to Option B (network learns `h(q)`, model
  output `sd = z - h(q)`, exact slope 1 in `z`, valid for any shape against
  a plane). To be formalised in spec 03.
- Dataset size: decided by a learning-curve study in spec 03 (Confirmed,
  user), e.g. 1e5 / 1e6 / 1e7 rows with mini-batch training. Needs approval.
- Accuracy tolerance (R01.15), sign-safety `DELTA_SIGN` and rates (R01.17),
  path-continuity tolerance (R01.18): set by the user after a baseline.
- Validation split by orientation (not by row); definition of "off-anchor".
- Numeric precision of the surrogate (float32 vs float64).
- C++ test details (spec 06): CMake+CTest is proposed in R06.4.
- Optional later: export plain weights for LibTorch-free C++ evaluation
  (user open to exploring; not needed now since LibTorch works).
- Review `inspired_codes/` later (including `inspired_codes/CPP_test/`,
  which still contains paths from the machine where it was first written).

## Deferred features

- Contact point and contact normal outputs.
- Other interactions (`ss_pyramid-wall`, `ss_cube-ss_cube`).
- LIGGGHTS integration (separate project).

## Relevant paths and commands

- Python env: `../env_folder/bin/python` (next to the project folder).
- LibTorch: `../libtorch/` (next to the project folder).
- Unit tests: `../env_folder/bin/python -m unittest discover tests` from the
  project root.

## Recent changes


- 2026-10-03: Spec 01 revision 3 (approved): R01.11-R01.13 no unseen inputs
  (gap safety zone up to `SD_COVER_MAX`), R01.16 convex-hull generalisation
  for any shape, new R01.17 sign safety, R01.18 continuity (canonical
  quaternion jumps 3 times on a 20000-step tumbling path). Code: shape
  constants, `classify_zone`, `support_height`. 54 tests, smoke 17/17.

- 2026-10-03: Renamed `constitution.md` to `CONSTITUTION.md` (capitalised like `AGENTS.md`, `MEMORY.md`, to match the user's other projects and central custom commands). References updated in README, AGENTS, spec 00, `check_structure.py`, `tests/test_structure.py`.
- 2026-10-02: Project scaffold created. Specs 00-06 written, reviewed, and revised.
- 2026-10-02: Git repo initialized and pushed to GitHub (`master`).
- 2026-10-02: Spec 00 revised: added `inspired_codes/`, clarified R00.4-R00.6, removed numbering scheme, added edge-case and validation requirements.
- 2026-10-02: Sign convention changed to LIGGGHTS signed distance. Updated constitution, AGENTS, MEMORY, specs 01-04 and 06, `cube_wall.py`, and tests.
- 2026-10-03: Spec 01 revision 2 completed (requirements accepted to proceed, design, tasks, tests, code, smoke test). Approved by user. Earlier draft note: rotation sense, qw=0 tie-break, invalid-input rejection, support-function definition, edge-length rescale, range [-0.1,-1e-5], invariances, accuracy and testbed requirements. design.md and tasks.md intentionally not yet updated.
- 2026-10-03: Governance update. Recorded end goal (LIGGGHTS runtime shape type), overlap-accuracy priority, deferred contact point/normal. Removed external-project paths. Added working-style rules 7-9 to AGENTS. Logged spec-review findings as items under revision.
