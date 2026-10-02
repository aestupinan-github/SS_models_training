# Spec 00 — Project structure and workflow

## Requirements

### R00.1 Top-level layout

`SS_models_training/` contains:

- `constitution.md` — durable principles
- `AGENTS.md` — working style and pointers
- `MEMORY.md` — confirmed facts, decisions, open questions
- `README.md` — project overview
- `interactions/` — one subfolder per interaction
- `CPP_test/` — shared standalone C++/LibTorch inference test area
- `specs/` — specifications

### R00.2 Interaction folder naming

- Shape types: `ss_<descriptor>` (e.g. `ss_cube`, `ss_pyramid`).
- Interaction folders: `ss_<shape>-wall` or `ss_<shape>-ss_<shape>`, joined with a hyphen.
- No constraints on the descriptor beyond the `ss_` prefix.

### R00.3 Interaction folder contents

Each interaction folder contains:

- `README.md` — interaction-specific overview and status
- `python/` — data generator, training, evaluation scripts
- `data/` — generated datasets, provenance records, fixed test cases
- `output/` — trained models, scalers, configs, logs, evaluation results
- `figures/` — diagnostic and evaluation figures

### R00.4 No cross-interaction mixing

Code, data, and outputs never reference or depend on another interaction's folder.

### R00.5 Self-contained interactions

Each interaction is fully self-contained. No shared utility library across interactions.

### R00.6 Adding a new interaction

1. Create a new folder under `interactions/` named per R00.2.
2. Copy the scaffold structure from R00.3.
3. Write its own specs under `specs/`.
4. Use the cube-wall scripts as a reference template (copy and adapt, not import).
5. No changes to existing interactions.

### R00.7 Git workflow

- Git repository lives at `SS_models_training/`.
- Commits only when explicitly requested by the user.
- `.gitignore` covers generated data, model artifacts, logs, figures, and C++ build output.

### R00.8 Specs organization

- Specs live under `specs/`.
- Each spec folder contains `requirements.md`, `design.md`, and `tasks.md`.
- Numbering: `00` project-level, `01–04` first interaction, `05` future interactions, `06` C++ test.
