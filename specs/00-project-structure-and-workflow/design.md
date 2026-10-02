# Spec 00 — Design

## Folder responsibilities

| Path | Responsibility |
|------|---------------|
| `interactions/<name>/python/` | Generator, training, evaluation scripts |
| `interactions/<name>/data/` | Datasets, provenance records, fixed test cases |
| `interactions/<name>/output/` | Models, scalers, configs, logs, results |
| `interactions/<name>/figures/` | Diagnostic and evaluation figures |
| `inspired_codes/` | Reference and prior-art scripts for inspiration |
| `CPP_test/` | Shared C++/LibTorch inference test (spec 06) |
| `specs/` | Specifications |

## Artifact organization

- Generated data and model artifacts are ignored by git (see `.gitignore`).
- Figures directories contain a `.gitkeep` to preserve the directory in git.
- Output artifacts are organized in versioned run folders (`run_001/`, `run_002/`) to prevent silent overwrites.

## Adding new interactions

New interactions copy the scaffold structure and write their own scripts. Prior art in `inspired_codes/` and existing interaction scripts serve as inspiration. Each interaction is self-contained with no cross-interaction runtime dependencies.
