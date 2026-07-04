# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-04

### Added
- CI via GitHub Actions (tests, pre-commit, conventional-commit PR check).
- `Objective.set_penalty_fn` for swapping the penalty function post-construction.
- Aux diagnostics: `power_thresholds`, `_supports_power_penalty` opt-in, and auto-logging of aux from standard eval methods when save tokens are enabled.
- `max_evals` and `max_time` as public properties.
- `checkpoint_format` and `checkpoint_dir` as user-facing storage knobs.
- Reconstructive `ProblemSpec` contract with `validate_spec_round_trip` and round-trip tests.
- `RunState` invariant contract and validation gate.
- Storage & Checkpointing reference docs page.

### Changed
- Refactored storage layer: untangled from problem layer, split submitter/organizer namespace, added typed `ProblemSpec` envelope.
- `Objective` save flags refactored into declarative `SaveConfig` with a token list; save policy moved onto `CheckpointManager.tick()`.
- Save tokens renamed to singular (`batched_loss`, `batched_grad`, `batched_hessian`, `batched_param`).
- Cleaned `Objective` constructor; eliminated all `obj._*` private access from algorithm and display layers.
- Renamed `problem_objective` to `objective`; removed sigmoid objective interface.
- Pinned Python to `>=3.11,<3.14`; pinned `pytest==9.0.3` for `pytest-cases` compatibility.
- Algorithm types aligned with folder structure; BO modularized.

### Fixed
- `BotorchTuRBO` unpacking of `_generate_batch` return value.
- Storage layer: unified `eval_type_counts`, synced serializer extension, removed redundant checkpoint check.
- PyCMA warmup indentation.
- Windows `TclError`.
- Restored `weighted_acq.py` stub required by `gp.py` import.
- Re-added import tests.

### Documentation
- New/updated docs for Objective API, Architecture, Storage, ProblemSpec contract, penalty swap, aux diagnostics, save tokens, `best_is_feasible`, and BO.

## [0.1.1] - previous PyPI release
