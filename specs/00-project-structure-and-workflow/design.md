# Spec 00 — Design

## Folder responsibilities

| Path | Responsibility |
|------|---------------|
| `interactions/<name>/python/` | Generator, training, evaluation scripts |
| `interactions/<name>/data/` | Datasets, provenance records, fixed test cases |
| `interactions/<name>/output/` | Models, scalers, configs, logs, results |
| `interactions/<name>/figures/` | Diagnostic and evaluation figures |
| `CPP_test/` | Shared C++/LibTorch inference test (spec 06) |
| `specs/` | Specifications |

## Artifact organization

- Generated data and model artifacts are ignored by git (see `.gitignore`).
- Figures directories contain a `.gitkeep` to preserve the directory in git.
- Output artifacts are organized in versioned run folders (`run_001/`, `run_002/`) to prevent silent overwrites.

## Adding new interactions

The cube-wall interaction serves as the reference template. New interactions copy the scaffold and adapt the scripts. Each interaction is self-contained with no cross-interaction dependencies.
