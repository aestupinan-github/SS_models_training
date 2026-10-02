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
- Model input: `[position_z, qw, qx, qy, qz]` (5 features).
- Model output: signed distance only (single scalar, no contact head).
- Each model is configuration-specific: one fixed shape, one fixed plane.
- Train at canonical size L = 1 (edge length 1, half-size 0.5). At query
  time, scale input z and output signed distance by the particle size.
  (The exact rescale rule is under revision, see open questions.)
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
- Phase 1 started: spec 01 tasks T01.1-T01.9 done (reference solution and
  unit tests). T01.10 (contract smoke tests) not done. No formal record of
  user approval of spec 01.
- Spec 02 (data generation), 03 (training), 04 (evaluation), 06 (C++) not
  started.
- Spec review in progress (2026-10-03): the specs are being re-reviewed for
  ambiguities before further implementation. Review findings are listed
  below as items under revision.

## Decisions and rationale

- Single-output model (signed distance only): simplifies the contract.
  Contact point/normal are deferred (user priority: overlap accuracy).
- Sampling: penetration log-uniform in `[-0.5, -1e-5]`; gap uniform in
  `(0, 0.1]`. (Under review: gap sampling, sample counts, critical set.)
- Model output clamped to `[-0.5, 0.1]`.
- Fresh C++ development, independent of `inspired_codes/`.

## Items under revision (found by spec review, 2026-10-03)

- **Target transform.** `t(x) = sign(x) log10(|x| + EPS)` is not invertible
  for `|x| < 1` and jumps from about +15 to about -15 across contact. Spec 03
  (and spec 06 step 9, the `eps` scaler field) must be replaced with an
  invertible, monotone transform.
- **Rescale rule.** Specs say divide z by the half-size. Canonical half-size
  is 0.5, so this is wrong by a factor 2 for the canonical cube. The rule
  must use the edge length L (= 2 x half-size).
- **Quaternion convention.** `cube_wall.py` rotates body to world (active).
  Spec must state this. Canonicalization needs a tie-break at `qw = 0`
  (e.g. first nonzero component positive) and must state normalization and
  invalid-input behavior.
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
- Should the network learn the full signed distance or only `z_touch(q)`
  with `sd = z - g(q)`? Decide in the spec 03 revision (accuracy first).
- Accuracy tolerances for the overlap (absolute `deltan` error near contact,
  relative error for deep penetration). Not yet specified.
- Validation split by orientation (not by row); definition of "off-anchor".
- Numeric precision (float32 vs float64) and out-of-range behavior.
- C++ test details (spec 06): CMake+CTest is proposed in R06.4.
- Review `inspired_codes/` later (including `inspired_codes/CPP_test/`,
  which still contains paths from the machine where it was first written).
- Approval of spec 01 to be confirmed by the user.

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

- 2026-10-02: Project scaffold created. Specs 00-06 written, reviewed, and revised.
- 2026-10-02: Git repo initialized and pushed to GitHub (`master`).
- 2026-10-02: Spec 00 revised: added `inspired_codes/`, clarified R00.4-R00.6, removed numbering scheme, added edge-case and validation requirements.
- 2026-10-02: Sign convention changed to LIGGGHTS signed distance. Updated constitution, AGENTS, MEMORY, specs 01-04 and 06, `cube_wall.py`, and tests.
- 2026-10-03: Governance update. Recorded end goal (LIGGGHTS runtime shape type), overlap-accuracy priority, deferred contact point/normal. Removed external-project paths. Added working-style rules 7-9 to AGENTS. Logged spec-review findings as items under revision.
