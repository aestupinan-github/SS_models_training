# Spec 00 — Project structure and workflow

## Requirements

### R00.1 Top-level layout

`SS_models_training/` contains:

- `CONSTITUTION.md` — durable principles
- `AGENTS.md` — working style and pointers
- `MEMORY.md` — confirmed facts, decisions, open questions
- `README.md` — project overview
- `interactions/` — one subfolder per interaction
- `inspired_codes/` — reference and prior-art scripts for inspiration (not imported at runtime)
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

Empty subfolders may be omitted. A `.gitkeep` is used to preserve empty directories in git.

### R00.4 No cross-interaction mixing

Code, data, and outputs never reference or depend on another interaction's folder at runtime. Reading prior art from `inspired_codes/` or external references for inspiration is allowed.

### R00.5 Self-contained interactions

Each interaction is fully self-contained. No shared utility library across interactions. Copying code from another interaction or from `inspired_codes/` is allowed; importing at runtime is not.

### R00.6 Adding a new interaction

1. Create a new folder under `interactions/` named per R00.2.
2. Copy the scaffold structure from R00.3.
3. Write its own specs under `specs/`.
4. Use prior art and existing scripts for inspiration when writing new code.
5. No changes to existing interactions.

### R00.7 Git workflow

- Git repository lives at `SS_models_training/`.
- Commits only when explicitly requested by the user.
- `.gitignore` covers generated data, model artifacts, logs, figures, and C++ build output.

### R00.8 Specs organization

- Specs live under `specs/`.
- Each spec folder contains `requirements.md`, `design.md`, and `tasks.md`.
- Specs are identified by name (string), not by a fixed numbering scheme.

### R00.9 Edge cases

- If an interaction folder name collides with an existing folder, the collision must be resolved before proceeding.
- Each spec folder must contain `requirements.md`, `design.md`, and `tasks.md`.

### R00.10 Validation

A structure-check script or documented validation step verifies that the folder structure conforms to this spec.
