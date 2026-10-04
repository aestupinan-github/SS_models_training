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
- Spec 02 (data generation): **requirements rev 2 APPROVED** (2026-10-03),
  with default fractions and `dataset_XXX` folders. Design and tasks rev 2
  APPROVED (2026-10-03). T02.1 done (config and validation), T02.2
  done (`shape_from_points`), T02.3 done (`uniform` and `special` orientation
  samplers with spin, yaw, canonicalization), T02.4 done (`near_kink` by
  bisection on tilt angle, exclusion by `u` in all samplers, D02.3), in
  `generate_dataset.py`. 135 tests OK.
  T02.5 done (`sample_orientations` mixture, `sample_sd`
  five parts with gap-safety fallback, `build_rows` with recomputed labels;
  D02.4 rounding/stratification decision). 165 tests OK.
  T02.6 done (`coverage_table`, `acceptance_checks` with 9 checks,
  `require_acceptance`; D02.5). T02.7 done (`generate_rows` with seed
  streams, `write_dataset`, `generate`: versioned `dataset_XXX` folders,
  never overwritten, `data.npz` + optional `data.csv`, `provenance.json`;
  D02.6; `.gitignore` covers dataset folders). 212 tests OK.
  **Next: T02.8** (CLI `python generate_dataset.py --n-orient N --k K
  --seed S [--csv] [--allow-full-scale] [--exclude FILE --exclude-angle
  DEG]`, printed summary, then ONE smoke dataset of at most 1e4 rows,
  `.npz` + CSV, in `interactions/ss_cube-wall/data/`, listed in the
  interaction README, reported as a smoke test). T02.8 not started; waits
  for the user's go. No dataset has been written to `data/` yet (tests use
  temporary folders only).
- Spec 03, 04, 06: still revision 1, to be revised.
- Pending: spec 03 (replace transform, Option B, orientation-based split,
  learning-curve study, sign-invariant input, mini-batch training), spec 04
  (off-anchor definition, invariance tests, tumbling-path test for
  continuity and sign safety), spec 06, `.gitignore`/`.gitkeep`.

## Decisions and rationale

- Spec 02 design decisions delegated to the agent by the user ("you
  decide, but record this", 2026-10-03), recorded in spec 02 `design.md`:
  D02.1 keep extra columns `sd_part`, `delta`; D02.2 `.gitignore` ignores
  `interactions/*/data/dataset_*/`; D02.3 held-out exclusion by body-frame
  direction `u` (superset of per-orientation exclusion, measured), symmetric
  images listed explicitly by spec 04.
- Agent decisions made during implementation, within the approved design,
  recorded in spec 02 `design.md`: D02.4 (T02.5) half-up rounding;
  stratified sd parts only if every positive part gets a row per
  orientation, else seeded random parts (defaults need K >= 50). D02.5
  (T02.6) extra `h_consistent` check; ranges checked per sd part; decade
  boundaries lower-inclusive, last decade includes 0.1; coverage n_fail
  counts failing cells. D02.6 (T02.7) two seed streams (`orientations`,
  `sd`) instead of the four listed in the design; safe publish via
  `os.mkdir` claim because `os.rename` silently replaces an empty
  directory (measured); CSV `%.17g` with names for codes; exclusion list in
  provenance by count and sha256.

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

- 2026-10-03: T02.7 implemented and committed: writer, versioned dataset folders (never overwritten), npz + CSV (17 digits, exact round trip), provenance with all R02.10 fields, SeedSequence streams, `.gitignore` for dataset folders. D02.6 recorded. 212 tests OK.
- 2026-10-03: T02.6 implemented and committed (code and tests `1532a4e`; design/tasks `1599b94`; this MEMORY update in the next commit): coverage table and 9 acceptance checks (each fails on a corrupted row; all pass on smoke data, worst label error 6.7e-16). D02.4 and D02.5 added to "Decisions and rationale". 191 tests OK.
- 2026-10-03: T02.5 implemented and committed: orientation mixture, sd sampler (5 parts, gap-safety fallback), row assembly with recomputed labels (0.0 error vs spec 01 reference on 10000 rows). D02.4 recorded. 165 tests OK.

- 2026-10-03: T02.4 implemented and committed: `sample_near_kink` (worst relative Delta error 6.5e-10 on the cube incl. quaternion chain; 1e5 rows 0.74 s), `exclusion_mask`, exclusion in all samplers (raises instead of looping when impossible). Two test expectations corrected and recorded in tasks.md and design.md (edge-like definition; tetrahedron tolerance). 135 tests OK.

- 2026-10-03: T02.3 implemented and committed: `uniform` and `special` samplers, vectorized canonicalization (bitwise equal to spec 01), `u_from_quats`, `support_gap`, `rotation_from_u`. 121 tests OK. Test note: canonical form is idempotent only to 1e-15 (re-normalization), as in spec 01.

- 2026-10-03: T02.2 implemented: `Shape`, `shape_from_points`, `ShapeError` (cube 6 facets, 12 edges, constants equal `cube_wall`; also tetrahedron and non-convex L-prism). 105 tests OK.

- 2026-10-03: T02.1 implemented: `GenConfig`, `validate_config`, `ConfigError`, approved default fractions, smoke limit 1e4 rows. 83 tests OK.

- 2026-10-03: Spec 02 requirements rev 2 approved; design and tasks rev 2 drafted (near_kink by bisection on tilt angle, measured exact to 2e-10; SeedSequence streams; atomic dataset_XXX folders; uncompressed npz, byte-reproducible).

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
