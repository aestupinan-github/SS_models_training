# SS_models_training

ML surrogate models of particle-shape contact interactions for DEM.

## Structure

```
SS_models_training/
  CONSTITUTION.md
  AGENTS.md
  MEMORY.md
  README.md
  interactions/
    ss_cube-wall/       # first interaction
      python/           # data generator, training, evaluation scripts
      data/             # generated datasets and provenance records
      output/           # trained models, scalers, configs, logs, results
      figures/
  CPP_test/             # future C++/LibTorch test area (spec 06, deferred)
  specs/                # specifications (under development)
```

## Interactions

| Interaction | Status |
|---|---|
| `ss_cube-wall` | Phase 1: spec 01 (contract) revised, implemented, approved |

## Getting started

See `AGENTS.md` for working style and `specs/` for specifications.
