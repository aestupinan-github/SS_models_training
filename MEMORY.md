# MEMORY.md

## Confirmed facts

- Project lives in `Programs/SS_models_training/`.
- Not a git repository (as of Phase 0 start).
- Model input: `[position_z, qw, qx, qy, qz]` (5 features).
- Model output: overlap only (single scalar, no contact head).
- Nondimensionalization: train at canonical size L = 1. At query time,
  divide input z by particle half-size and multiply output overlap back.
- LIGGGHTS sign convention: surrogate overlap is positive; `deltan = -overlap`.
- Each model is configuration-specific: one fixed shape, one fixed plane.
- Existing prior art: `XDEM/temp/Test_meterCube_PeppaClaudeSonet5_Strategy_*`
  (Python generator, training, evaluation) and `XDEM/Cases/CPP_test/` (C++/
  LibTorch inference tests). These are reference only; the new project is
  fresh development.

## Unverified prior context

- Earlier generators used anchor heights with jitter and decade-stratified
  or continuous sampling. Model trained on log10 overlap target with EPS
  offset and scaler. Exported as TorchScript with ASCII scaler file.
- Earlier validation used off-anchor slope check and analytical vs surrogate
  vs C++ comparisons.
- These are design options to review, not fixed decisions.

## Current scope and status

- Phase 0: scaffold created (constitution, AGENTS, MEMORY, README,
  interactions/ss_cube-wall/, CPP_test/). Specs not yet written — under
  discussion.
- No generator, training, or C++ code written yet.

## Decisions and rationale

- Single-output model (overlap only): simplifies the contract; contact
  classification can be added later if needed.
- Fresh C++ development inspired by `XDEM/Cases/CPP_test/` but independent.
- Git repo to be initialized in `SS_models_training/`.

## Open questions

- Model input: 5-input `[z, qw, qx, qy, qz]` confirmed by user.
- C++ test: CMake+CTest or standalone runner? To be decided in spec 06.
- `.gitignore` contents for generated data and model artifacts.
- Spec list and scope to be finalized before writing.

## Relevant paths and commands

- Prior art: `XDEM/temp/Test_meterCube_PeppaClaudeSonet5_Strategy_FullContinuous_v6.1/`
- Prior art: `XDEM/Cases/CPP_test/`
- Python env: `env_folder/bin/python`
- LibTorch: `libtorch/`

## Recent changes

- 2026-10-02: Project scaffold created. Specs 00–06 written and reviewed.
