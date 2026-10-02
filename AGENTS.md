# AGENTS.md

## Project purpose

Build and train ML surrogate models of particle-shape contact interactions
for DEM. First target: `ss_cube-wall`. The structure must allow more
interactions later (e.g. `ss_pyramid-wall`, `ss_cube-ss_cube`).

**End goal (Confirmed, user).** A new LIGGGHTS particle-shape type whose
overlap is computed by a trained surrogate, evaluated from C++ at runtime
(no Python). The surrogate replaces the overlap computation for that shape.
The LIGGGHTS integration lives in a separate project and is out of scope
here; this repo delivers trained artifacts and their interface contract.

**Priority (Confirmed, user).** Accuracy of the overlap (signed distance) comes
first. Contact point and contact normal are a deferred later feature.

## Phases

- **Phase 0:** Setup and specs only. Scaffold, constitution, memory, specs.
- **Phase 1:** Data generator for `ss_cube-wall` (after spec 01 approval).
- **Phase 2:** Training script and artifact export (after generator review).
- **Phase 3:** Evaluation.
- **Later:** C++ test and other interactions. Contact point/normal outputs.

Do not start a later phase without explicit request.

## Working style

1. Inspect before deciding. Read actual files first.
2. Ask, don't guess. If a decision affects structure, the data contract,
   scientific interpretation, or scope and the repo cannot settle it, stop
   and ask.
3. Label statements as *Confirmed*, *Assumption*, *Proposal*, or *Open
   question*.
4. Preserve existing work. No destructive changes without approval.
5. Cost control. No full-scale generation, training, commit, or push without
   approval. Smoke tests allowed after phase approval, reported as such.
6. Update `MEMORY.md` as you go.
7. Facts stated by the user are *Confirmed (user)*. Do not re-verify them.
8. Do not search for external codebases, other projects, or the web unless
   the user asks. `inspired_codes/` is reference only.
9. Requirement changes go in order: `requirements.md`, `design.md`,
   `tasks.md`, then code, then tests. Each step needs user approval.

## Naming convention

- Surrogate shapes: `ss_<descriptor>` (e.g. `ss_cube`).
- Interaction folders join partners with a hyphen: `ss_cube-wall`,
  `ss_cube-ss_cube`, `ss_pyramid-wall`.
- `ss_cube` is the shape type; `ss_cube-wall` is the interaction and folder.

## Sign convention

- Model predicts the signed distance between shape and wall (for a polyhedron
  and a plane: the lowest vertex height above the plane).
- Positive = gap (no contact), negative = penetration.
- LIGGGHTS `deltan` is negative for actual overlap, so
  `deltan = signed_distance` (Confirmed, user).

## Pointers

- `CONSTITUTION.md` — durable principles.
- `MEMORY.md` — confirmed facts, open questions, decisions, recent changes.
- `specs/` — specifications per phase and interaction.
- `interactions/` — one folder per interaction.
- `CPP_test/` — future C++/LibTorch test area (spec 06, deferred).
- `inspired_codes/` — prior reference scripts, inspiration only, partly
  unfinished; to be reviewed later.
