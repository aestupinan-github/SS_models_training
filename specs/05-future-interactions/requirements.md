# Spec 05 — Future interactions

## Requirements

### R05.1 New interaction = new folder

- Create a new folder under `interactions/` named per convention (`ss_<shape>-wall` or `ss_<shape>-ss_<shape>`).

### R05.2 Self-contained

- Each interaction has its own `python/`, `data/`, `output/`, `figures/`, and its own specs under `specs/`.

### R05.3 No cross-interaction dependencies

- Code, data, and outputs never reference another interaction.

### R05.4 Reuse patterns, not code

- The cube-wall generator/training/evaluation scripts serve as reference templates.
- Each interaction writes its own scripts (copy and adapt, not import).

### R05.5 New specs

- Each new interaction gets its own spec folder (e.g. `07-ss_pyramid-wall-interaction-contract/`).

### R05.6 Shared C++ test

- All interactions use the shared `CPP_test/` area (spec 06).

### R05.7 Shape-shape interactions

- Forward-looking: shape-shape interactions (e.g. `ss_cube-ss_cube`) will need a different contract structure (two poses instead of pose + fixed plane).
- Details to be discovered when the first shape-shape interaction is requested.
