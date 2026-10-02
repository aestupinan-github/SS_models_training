# Spec 02 — Tasks

## Status

**DRAFT, revision 2, 2026-10-03. Not approved.** Revision 1 tasks
(T02.1-T02.9, none started) are replaced. Each task: tests first, see them
fail, implement, run all tests.

- [ ] **T02.1 Configuration and validation.** R02.11, R02.13
  - Files: `generate_dataset.py` (`GenConfig`, `validate_config`, `ConfigError`), `tests/test_generate_dataset.py`.
  - Done when: each invalid case of R02.11 and the size limit raise `ConfigError`; a valid default config passes.
- [ ] **T02.2 Shape description from a point set.** R02.1, R02.3
  - Done when: cube gives 6 facets, 12 edges, `R_CIRC` and `H_MIN` equal to `cube_wall` constants.
- [ ] **T02.3 Orientation sampler: `uniform` and `special`, with spin, yaw and canonicalization.** R02.3
  - Done when: special `Δ` values and `h` values are exact; quaternions canonical; yaw statistics within tolerance.
- [ ] **T02.4 Orientation sampler: `near_kink` by bisection, and exclusion.** R02.3
  - Done when: achieved `Δ` within `1e-9` relative of target; every decade of `[1e-6, 1e-1]` populated; no orientation within the exclusion angle.
- [ ] **T02.5 `sd` sampler and row assembly with recomputed labels.** R02.1, R02.2, R02.4, R02.6
  - Done when: ranges per part hold; gap-safety fallback works; labels match reference within `1e-12`; columns, dtypes and order correct.
- [ ] **T02.6 Coverage table and acceptance checks.** R02.5, R02.12
  - Done when: checks pass on generated rows and fail on a deliberately corrupted row.
- [ ] **T02.7 Writer, versioning, provenance, seeding.** R02.7, R02.8, R02.9, R02.10
  - Done when: new `dataset_XXX` folder per run, never overwritten; same seed gives identical sha256; provenance holds all R02.10 fields; CSV round trip at 17 digits; `.gitignore` covers dataset folders.
- [ ] **T02.8 CLI, summary, smoke dataset.** R02.14, completion criteria
  - Done when: one smoke dataset (at most 1e4 rows, `.npz` + CSV) is generated in `data/`, all checks pass, it is listed in the interaction README, reported as a smoke test.
- [ ] **T02.9 Full-scale generation.** R02.13
  - Only after user approval, with sizes from the spec 03 learning-curve study.
