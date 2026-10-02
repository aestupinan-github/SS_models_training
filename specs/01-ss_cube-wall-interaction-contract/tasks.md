# Spec 01 — Tasks

## Status

Revision 2, 2026-10-03. All tasks done; spec approved by the user (2026-10-03). T01.1-T01.9 were completed under revision 1 and are
kept as history. T01.10 is replaced by the tasks below.

### Completed under revision 1

- [x] T01.1 Define cube geometry and size convention
- [x] T01.2 Define wall geometry and coordinate frame
- [x] T01.3 Define quaternion order and canonicalization
- [x] T01.4 Define signed-distance computation (reference solution)
- [x] T01.5 Define model input/output semantics
- [x] T01.6 Define scaling convention
- [x] T01.7 Define sign convention (LIGGGHTS)
- [x] T01.8 Define signed-distance and position ranges
- [x] T01.9 Implement reference solution (`cube_wall.py`) and unit tests. Covers R01.1, R01.3, R01.4 (partial), R01.5, R01.12

### Revision 2

- [x] **T01.10 Revise quaternion handling.** R01.4
  - Tests first: `qw == 0` pair maps to one output; non-unit normalized; zero and NaN rejected with `ValueError`.
  - Done when: new tests fail before the change and pass after; existing tests still pass.
- [x] **T01.11 Rescale and range helpers.** R01.7, R01.13
  - Tests first: `to_canonical` / `from_canonical` round trip and `L_actual` validation; `classify_sd` boundaries; `is_far` never contradicts exact `sd > 0.1` on random poses.
  - Done when: tests pass.
- [x] **T01.12 Invariance tests.** R01.14
  - Done when: tests for slope, yaw, 24 symmetries, `q`/`-q`, and `0.5 <= g <= sqrt(3)/2` pass on random poses with a fixed seed.
- [x] **T01.13 Contract smoke test script.** R01.4, R01.5, R01.7, R01.13, R01.14
  - Done when: `python smoke_contract.py` with a fixed seed prints pass/fail counts labelled as a smoke test, exits non-zero on failure, and is reported as a smoke test, not a result.
- [x] **T01.14 Documentation update.** R01.*
  - Done when: `interactions/ss_cube-wall/README.md` and `MEMORY.md` reflect the revised contract.
