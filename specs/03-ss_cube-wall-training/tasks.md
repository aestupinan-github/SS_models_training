# Spec 03 — Tasks

## Status

**DRAFT, revision 2, 2026-10-04 (amended for `scalers.dat`, S03.10). Not approved.** Revision 1 tasks
(T03.1-T03.12, none started) are replaced. Each task: tests first, see them
fail, implement, run all tests, commit, update MEMORY.md, stop.

- [ ] **T03.1 Fixed input layer and network.** R03.1, R03.3, R03.14
  - Files: `surrogate_model.py` (`UFeature`, `HNet`), `tests/test_surrogate_model.py`.
  - Done when: `u(q)` matches scipy's `-R^T e_z` within 1e-15; `q` and `-q` give bit-identical `u`; `HNet` builds from a hidden-size tuple and activation name.
- [ ] **T03.2 Exported model with status codes, and `scalers.dat`.** R03.2, R03.4, R03.9, R03.16
  - Files: `surrogate_model.py` (`SurrogateModel`, status constants, `export`, `write_scalers`, `read_scalers`).
  - Done when: float64 `(N, 5) -> (sd, status)`; `sd = z - h_pred`; `far`, `too_deep`, `invalid` cases and mixed batches as specified; TorchScript save and load round trip; outputs equal before and after loading; `scalers.dat` values equal the model buffers and constants exactly and are read by a parser that mimics the reference loader (D03.1).
- [ ] **T03.3 Config, dataset loader, split.** R03.5, R03.6, R03.12, R03.13, R03.14
  - Files: `train_surrogate.py` (`TrainConfig`, `validate_train_config`, `load_dataset`, `split_by_orientation`), `tests/test_train_surrogate.py`.
  - Done when: invalid configs and smoke-limit violations raise `TrainConfigError`; checksum mismatch, missing file or column raise `DatasetError`; split is disjoint by orientation, seeded and reproducible.
- [ ] **T03.4 Training loop and reproducibility.** R03.7, R03.10, R03.13
  - Done when: on a tiny synthetic dataset the validation loss decreases; early stopping and wall-time stop work; non-finite loss raises `TrainingError`; two CPU runs with the same seed give identical weights.
- [ ] **T03.5 Metrics incl. tumbling path.** R03.8
  - Done when: each metric returns the expected value on a constructed prediction; `q`/`-q` metric is 0 for the model; tumbling-path metric is reproducible.
- [ ] **T03.6 Run writer, run record, `train()`, CLI.** R03.11, R03.13, R03.15
  - Done when: new `run_XXX` per call, never overwritten; `run_XXX` holds `model.pt`, `state_dict.pth`, `scalers.dat`, `run.json`; `run.json` holds all R03.11 fields and correct checksums of all three artifacts; failures leave no complete-looking folder; CLI exit codes 0/1/2; `.gitignore` covers `output/run_*/`.
- [ ] **T03.7 Smoke training run on `dataset_001`.** completion criteria
  - Done when: one run (at most 1e4 rows, at most 10 min) writes `output/run_001`; exported model passes the interface tests when loaded from disk; metrics printed and recorded, labelled as a smoke test; run listed in the interaction README.
