"""Tests for the modular dfbench.core.storage package."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dfbench.core.storage import (
    CheckpointManager,
    JsonCheckpointSerializer,
    LocalFilesystemBackend,
    NpzCheckpointSerializer,
    NpzRunCollectionSerializer,
    RunDataExporter,
    RunMetadata,
    RunPathResolver,
    RunState,
    RunStateValidationException,
    validate_run_state,
)
from dfbench.core.storage.backends import StorageBackend
from dfbench.core.storage.serializers import CheckpointSerializer


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _make_state(eval_count: int = 5) -> RunState:
    return RunState(
        loss_history=np.array([1.0, 0.5, 0.3, 0.2, 0.1][:eval_count], dtype=object),
        grad_history=np.array(
            [np.array([0.1, 0.2]) for _ in range(eval_count)], dtype=object
        ),
        hessian_history=np.array([np.eye(2) for _ in range(eval_count)], dtype=object),
        params_history=np.array(
            [np.array([0.0, 0.0]) for _ in range(eval_count)], dtype=object
        ),
        eval_type_history=np.array([1] * eval_count, dtype=object),
        time_steps=np.array([0.0, 0.1, 0.2, 0.3, 0.4][:eval_count], dtype=object),
        eval_count=eval_count,
        best_loss=0.1,
        best_params=np.array([0.0, 0.0], dtype=np.float64),
        improvement_count=eval_count,
        evals_since_improvement=0,
        log_call_count=eval_count,
        eval_type_counts={1: eval_count},
        metadata=RunMetadata(
            problem_name="test_problem",
            algorithm_name="test_algo",
            hyper_param_str="lr0.1",
            timestamp="2026-01-01_00-00-00",
            max_time=100.0,
            max_evals=1000,
            unbounded=False,
        ),
    )


# ---------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------


class TestRunMetadata:
    def test_round_trip(self):
        m = RunMetadata(problem_name="p", algorithm_name="a", timestamp="t")
        d = m.to_dict()
        assert d["format_version"] == 1
        m2 = RunMetadata.from_dict(d)
        assert m2.problem_name == "p"
        assert m2.algorithm_name == "a"
        assert m2.timestamp == "t"

    def test_future_version_rejected(self):
        d = RunMetadata().to_dict()
        d["format_version"] = 999
        with pytest.raises(ValueError, match="newer"):
            RunMetadata.from_dict(d)


# ---------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------


@pytest.fixture(params=["npz", "json"])
def serializer(request) -> CheckpointSerializer:
    return {"npz": NpzCheckpointSerializer(), "json": JsonCheckpointSerializer()}[
        request.param
    ]


class TestSerializers:
    def test_round_trip(self, serializer):
        state = _make_state()
        data = serializer.serialize(state)
        assert isinstance(data, bytes)
        out = serializer.deserialize(data)
        assert out.eval_count == state.eval_count
        assert out.best_loss == pytest.approx(state.best_loss)
        np.testing.assert_allclose(out.best_params, state.best_params)
        assert out.metadata.problem_name == state.metadata.problem_name
        assert out.eval_type_counts == state.eval_type_counts

    def test_empty_best_params(self, serializer):
        state = _make_state()
        state.best_params = np.array([], dtype=np.float64)
        out = serializer.deserialize(serializer.serialize(state))
        assert out.best_params.size == 0

    def test_legacy_missing_keys(self):
        # A minimal NPZ without metadata/version; must still load.
        import io

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            loss_history=np.array([1.0, 2.0], dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array([], dtype=object),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.array([0.0, 0.1], dtype=object),
            eval_count=np.int64(2),
            best_loss=np.float64(1.0),
            best_params=np.array([], dtype=np.float64),
            improvement_count=np.int64(0),
            evals_since_improvement=np.int64(0),
        )
        out = NpzCheckpointSerializer().deserialize(buf.getvalue())
        assert out.eval_count == 2
        assert out.metadata.problem_name == "problem"  # default


# ---------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------


class TestLocalFilesystemBackend:
    def test_save_load_exists_delete(self, tmp_path):
        backend = LocalFilesystemBackend(root=tmp_path)
        key = "sub/dir/file.bin"
        assert not backend.exists(key)
        backend.save_bytes(key, b"hello")
        assert backend.exists(key)
        assert backend.load_bytes(key) == b"hello"
        backend.delete(key)
        assert not backend.exists(key)

    def test_is_storage_backend_protocol(self):
        assert isinstance(LocalFilesystemBackend(), StorageBackend)

    def test_overwrite_is_atomic(self, tmp_path):
        backend = LocalFilesystemBackend(root=tmp_path)
        key = "file.bin"
        backend.save_bytes(key, b"first")
        backend.save_bytes(key, b"second")
        assert backend.load_bytes(key) == b"second"
        # No leftover temp files
        assert all(p.suffix != ".tmp" for p in tmp_path.iterdir())


# ---------------------------------------------------------------------
# RunPathResolver
# ---------------------------------------------------------------------


class TestRunPathResolver:
    def test_default_layout(self):
        r = RunPathResolver()
        p = r.checkpoint_path(
            problem_name="voyager",
            algorithm_name="adam/gd",
            timestamp="2026-01-01_00-00-00",
            hyper_param_str="lr0.1",
            max_time=100.0,
            max_evals=1000,
        )
        # Flat default: no algo dir, hp string lives in the filename.
        assert p.name == "voyager_adam_gd_lr0.1_2026-01-01_00-00-00.npz"
        assert "time100s_evals1000" in p.parts
        assert "adam_gd_lr0.1" not in p.parts

    def test_algo_directory_layout(self):
        r = RunPathResolver(algo_directory=True)
        p = r.checkpoint_path(
            problem_name="voyager",
            algorithm_name="adam/gd",
            timestamp="2026-01-01_00-00-00",
            hyper_param_str="lr0.1",
            max_time=100.0,
            max_evals=1000,
        )
        assert p.name == "voyager_adam_gd_lr0.1_2026-01-01_00-00-00.npz"
        assert "time100s_evals1000" in p.parts
        assert "adam_gd_lr0.1" in p.parts

    def test_algo_directory_without_hyperparam(self):
        r = RunPathResolver(algo_directory=True)
        p = r.checkpoint_path(
            problem_name="voyager",
            algorithm_name="adam_gd",
            timestamp="2026-01-01_00-00-00",
        )
        # No hp string: dir segment collapses to just the algo name.
        assert p.name == "voyager_adam_gd_2026-01-01_00-00-00.npz"
        assert "adam_gd" in p.parts

    def test_unlimited_budget(self):
        r = RunPathResolver()
        p = r.checkpoint_path("p", "a", "t")
        assert "unlimited" in p.parts


# ---------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_load_round_trip(self, tmp_path):
        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(),
        )
        state = _make_state()
        path = manager.save(state)
        assert path.exists()
        assert path.is_absolute()
        assert str(path).startswith(
            str(tmp_path.resolve())
        )  # backend joined its root onto the key
        assert manager.last_checkpoint_eval == state.eval_count

        loaded = manager.load(path)
        assert loaded.eval_count == state.eval_count
        # Cached path means a second save without overrides rewrites the same file
        path2 = manager.save(state)
        assert path2 == path

    def test_tick_skips_when_not_due(self, tmp_path):
        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(),
            save_every=5,
        )
        called = {"n": 0}

        def factory():
            called["n"] += 1
            return _make_state()

        # eval_count=3, save_every=5 -> not due, returns 0.0
        assert manager.tick(3, factory) == 0.0
        assert called["n"] == 0
        # eval_count=5 -> due, returns positive duration
        dt = manager.tick(5, factory)
        assert dt >= 0.0
        assert called["n"] == 1

    def test_explicit_path_overrides_resolver(self, tmp_path):
        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(),
        )
        state = _make_state()
        explicit = tmp_path / "custom.npz"
        path = manager.save(state, explicit_path=explicit)
        assert path.resolve() == explicit.resolve()
        assert explicit.exists()


class TestCheckpointManagerDefaultRoot:
    """Regression: the default relative ./data/objective_run_data root
    must round-trip without double prefixing. Before the fix, manager.save
    returned a path that did not exist, because the backend joined its
    root onto a path that already contained the root."""

    def test_default_relative_root_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = CheckpointManager()  # all defaults
        state = _make_state()
        path = manager.save(state)
        assert path.exists(), f"{path} should exist (was the double-prefix bug)"
        assert path.is_absolute()
        loaded = manager.load(path)
        assert loaded.eval_count == state.eval_count
        # Second save overwrites the cached path
        path2 = manager.save(state)
        assert path2 == path


# ---------------------------------------------------------------------
# RunDataExporter
# ---------------------------------------------------------------------


class TestRunDataExporter:
    def test_exports_json_and_png(self, tmp_path):
        exporter = RunDataExporter(root=str(tmp_path))
        state = _make_state()
        out_dir = exporter.export(state, problem=None, print_summary=False)
        assert out_dir.exists()
        files = list(out_dir.iterdir())
        assert any(p.suffix == ".json" for p in files)
        assert any(p.suffix == ".png" for p in files)

    def test_parameters_json_contents(self, tmp_path):
        exporter = RunDataExporter(root=str(tmp_path))
        state = _make_state()
        out_dir = exporter.export(state, problem=None, print_summary=False)
        params_files = list(out_dir.glob("*_parameters.json"))
        assert len(params_files) == 1
        content = json.loads(params_files[0].read_text())
        assert content == [0.0, 0.0]


# ---------------------------------------------------------------------
# RunCollectionSerializer (used by Benchmark)
# ---------------------------------------------------------------------


class TestNpzRunCollectionSerializer:
    def test_collection_round_trip(self):
        """Serialize/deserialize a collection of RunState objects."""
        ser = NpzRunCollectionSerializer()
        states = [_make_state(eval_count=3), _make_state(eval_count=5)]
        data = ser.serialize_collection(
            algorithm_name="adam",
            hyperparameters={"lr": 0.1},
            runs=states,
        )
        assert isinstance(data, bytes)

        name, hparams, loaded = ser.deserialize_collection(data)
        assert name == "adam"
        assert hparams == {"lr": 0.1}
        assert len(loaded) == 2
        assert loaded[0].eval_count == 3
        assert loaded[1].eval_count == 5
        # Metadata (including problem_spec) survives
        assert loaded[0].metadata.problem_name == "test_problem"

    def test_collection_empty_runs(self):
        ser = NpzRunCollectionSerializer()
        data = ser.serialize_collection("empty", {}, [])
        name, hparams, loaded = ser.deserialize_collection(data)
        assert name == "empty"
        assert loaded == []

    def test_extension_attribute(self):
        assert NpzRunCollectionSerializer().extension == "npz"


# ---------------------------------------------------------------------
# Serializer extension sync (#5)
# ---------------------------------------------------------------------


class TestSerializerExtensionSync:
    def test_json_serializer_produces_json_path(self, tmp_path):
        """CheckpointManager with JsonCheckpointSerializer uses .json paths."""
        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            serializer=JsonCheckpointSerializer(),
            resolver=RunPathResolver(),
        )
        assert manager.resolver.extension == "json"
        state = _make_state()
        path = manager.save(state)
        assert path.suffix == ".json"


# ---------------------------------------------------------------------
# RunState invariants (#2)
# ---------------------------------------------------------------------


def _mutate(state: RunState, **kw) -> RunState:
    """Return a shallow copy of ``state`` with the given fields replaced."""
    import dataclasses as dc

    return dc.replace(state, **kw)


class TestValidateRunState:
    # --- baseline -------------------------------------------------------

    def test_valid_state_passes_strict(self):
        report = validate_run_state(_make_state(), strict=True)
        assert report.ok
        assert report.errors == []

    def test_valid_state_passes_non_strict(self):
        report = validate_run_state(_make_state(), strict=False)
        assert report.ok

    def test_empty_state_is_valid(self):
        """eval_count==0, all histories empty, best_loss=inf is legal."""
        state = RunState(
            loss_history=np.array([], dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array([], dtype=object),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.array([], dtype=object),
            eval_count=0,
            best_loss=float("inf"),
            best_params=np.array([], dtype=np.float64),
            improvement_count=0,
            evals_since_improvement=0,
            log_call_count=0,
            eval_type_counts={},
            metadata=RunMetadata(),
        )
        report = validate_run_state(state)
        assert report.ok, [str(e) for e in report.errors]

    def test_legacy_reduced_state_is_valid(self):
        """RunData.to_run_state drops grad/hessian/eval_type to empty and
        zeroes improvement/log_call counters. That shape must validate."""
        state = RunState(
            loss_history=np.array([1.0, 2.0], dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array(
                [np.array([0.0, 0.0]), np.array([1.0, 1.0])], dtype=object
            ),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.array([0.0, 0.1], dtype=object),
            eval_count=2,
            best_loss=1.0,
            best_params=np.array([0.0, 0.0], dtype=np.float64),
            improvement_count=0,
            evals_since_improvement=0,
            log_call_count=0,
            eval_type_counts={},
            metadata=RunMetadata(),
        )
        report = validate_run_state(state)
        assert report.ok, [str(e) for e in report.errors]

    # --- Tier A: structural --------------------------------------------

    def test_A1_negative_eval_count(self):
        s = _mutate(_make_state(), eval_count=-1)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A1" for e in r.errors)

    def test_A1_float_eval_count(self):
        s = _mutate(_make_state(), eval_count=5.0)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A1" for e in r.errors)

    def test_A2_loss_history_wrong_type(self):
        s = _mutate(_make_state(), loss_history=[1.0, 0.5, 0.3, 0.2, 0.1])
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A2" for e in r.errors)

    def test_A2_ndarray_histories_are_allowed(self):
        """NumPy collapses uniform-shape object arrays to N-D numeric
        arrays (e.g. five (2,) gradients -> shape (5, 2)). The contract is
        "first axis is the time axis", not strictly 1-D, so this is legal."""
        s = _mutate(
            _make_state(),
            grad_history=np.zeros((5, 2), dtype=np.float64),
        )
        r = validate_run_state(s)
        # A2 must pass; the only error, if any, would be from elsewhere.
        assert all(e.invariant != "A2" for e in r.errors)

    def test_A3_history_length_mismatch(self):
        s = _mutate(
            _make_state(),
            time_steps=np.array(
                [0.0, 0.1, 0.2], dtype=object
            ),  # len 3, log_call_count 5
        )
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A3" for e in r.errors)

    def test_A3_batched_histories_align_to_log_call_count(self):
        """Batched calls add multiple evals but one history row."""
        state = RunState(
            loss_history=np.array(
                [np.array([2.0, 1.0]), 0.8, np.array([3.0, 4.0, 5.0])],
                dtype=object,
            ),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array(
                [
                    np.array([[0.0, 0.0], [0.1, 0.1]]),
                    np.array([0.2, 0.2]),
                    np.array([[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]),
                ],
                dtype=object,
            ),
            eval_type_history=np.array([5, 1, 5], dtype=object),
            time_steps=np.array([0.0, 0.1, 0.2], dtype=object),
            eval_count=6,
            best_loss=0.8,
            best_params=np.array([0.2, 0.2]),
            improvement_count=2,
            evals_since_improvement=0,
            log_call_count=3,
            eval_type_counts={5: 2, 1: 1},
            metadata=RunMetadata(),
        )

        r = validate_run_state(state)
        assert r.ok, [str(e) for e in r.errors]

    def test_A3_empty_history_is_allowed(self):
        """A disabled history (empty array) must not trip A3."""
        s = _mutate(
            _make_state(),
            grad_history=np.array([], dtype=object),
        )
        r = validate_run_state(s)
        assert r.ok, [str(e) for e in r.errors]

    def test_A4_best_params_2d(self):
        s = _mutate(
            _make_state(),
            best_params=np.zeros((2, 2), dtype=np.float64),
        )
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A4" for e in r.errors)

    def test_A5_eval_type_counts_non_int_key(self):
        s = _mutate(_make_state(), eval_type_counts={"1": 5})
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A5" for e in r.errors)

    def test_A5_eval_type_counts_negative_value(self):
        s = _mutate(_make_state(), eval_type_counts={1: -5})
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A5" for e in r.errors)

    def test_A5_eval_type_counts_wrong_type(self):
        s = _mutate(_make_state(), eval_type_counts=None)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A5" for e in r.errors)

    def test_A6_metadata_not_runmetadata(self):
        s = _mutate(_make_state(), metadata={"problem_name": "x"})
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A6" for e in r.errors)

    def test_A7_negative_improvement_count(self):
        s = _mutate(_make_state(), improvement_count=-1)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A7" for e in r.errors)

    def test_A7_negative_log_call_count(self):
        s = _mutate(_make_state(), log_call_count=-2)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "A7" for e in r.errors)

    # --- Tier B: semantic (skipped when strict=False) ------------------

    def test_B1_best_loss_inf_with_recorded_loss(self):
        """eval_count>0 with a real loss recorded but best_loss=inf is wrong."""
        s = _mutate(_make_state(eval_count=3), best_loss=float("inf"))
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "B1" for e in r.errors)

    def test_B1_best_loss_finite_without_recorded_loss(self):
        """best_loss finite but loss_history is all-NaN/empty is wrong."""
        s = RunState(
            loss_history=np.array([np.nan], dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array([], dtype=object),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.array([0.0], dtype=object),
            eval_count=1,
            best_loss=0.5,
            best_params=np.array([], dtype=np.float64),
            improvement_count=0,
            evals_since_improvement=0,
            log_call_count=0,
            eval_type_counts={},
            metadata=RunMetadata(),
        )
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "B1" for e in r.errors)

    def test_B1_inf_best_loss_legal_for_grad_only_evals(self):
        """A hessian-only / grad-only call increments eval_count and appends
        NaN to loss_history without updating best_loss. This is legal."""
        s = RunState(
            loss_history=np.array([np.nan], dtype=object),
            grad_history=np.array([np.array([0.1, 0.2])], dtype=object),
            hessian_history=np.array([np.eye(2)], dtype=object),
            params_history=np.array([np.array([0.0, 0.0])], dtype=object),
            eval_type_history=np.array([2], dtype=object),  # grad-only
            time_steps=np.array([0.0], dtype=object),
            eval_count=1,
            best_loss=float("inf"),
            best_params=np.array([], dtype=np.float64),
            improvement_count=0,
            evals_since_improvement=1,
            log_call_count=1,
            eval_type_counts={2: 1},
            metadata=RunMetadata(),
        )
        r = validate_run_state(s)
        assert r.ok, [str(e) for e in r.errors]

    def test_B2_best_loss_drift_is_the_headline_check(self):
        """best_loss=0.05 but nanmin(loss_history)=0.1 -> must be caught."""
        s = _mutate(_make_state(eval_count=5), best_loss=0.05)
        r = validate_run_state(s)
        assert not r.ok
        assert any(e.invariant == "B2" for e in r.errors), [str(e) for e in r.errors]

    def test_B2_handles_batched_loss_entries(self):
        """Batched loss entries reduce via nanmin before comparing."""
        s = RunState(
            loss_history=np.array(
                [np.array([1.0, 0.5, 0.3]), np.array([0.2, 0.1, 0.4])],
                dtype=object,
            ),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array([np.zeros(2), np.ones(2)], dtype=object),
            eval_type_history=np.array([5, 5], dtype=object),
            time_steps=np.array([0.0, 0.1], dtype=object),
            eval_count=6,
            best_loss=0.1,
            best_params=np.ones(2),
            improvement_count=0,
            evals_since_improvement=0,
            log_call_count=2,
            eval_type_counts={5: 2},
            metadata=RunMetadata(),
        )
        r = validate_run_state(s)
        assert r.ok, [str(e) for e in r.errors]

    def test_B4_call_count_drift(self):
        s = _mutate(_make_state(), eval_type_counts={1: 5, 3: 1})  # sum=6 != log=5
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "B4" for e in r.errors)

    def test_B5_improvement_exceeds_calls(self):
        s = _mutate(_make_state(), improvement_count=99)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "B5" for e in r.errors)

    def test_B6_stagnation_exceeds_evals(self):
        s = _mutate(_make_state(eval_count=5), evals_since_improvement=99)
        r = validate_run_state(s)
        assert not r.ok and any(e.invariant == "B6" for e in r.errors)

    # --- strict vs non-strict ------------------------------------------

    def test_strict_false_skips_b_tier(self):
        """A state failing only B2 must pass under strict=False."""
        s = _mutate(_make_state(eval_count=5), best_loss=0.05)  # B2 violation
        r_loose = validate_run_state(s, strict=False)
        assert r_loose.ok, [str(e) for e in r_loose.errors]
        r_strict = validate_run_state(s, strict=True)
        assert not r_strict.ok

    # --- aggregation ----------------------------------------------------

    def test_collects_multiple_errors(self):
        """A state failing both A3 and B2 reports both in one pass."""
        s = _mutate(
            _mutate(_make_state(eval_count=5), best_loss=0.05),  # B2
            time_steps=np.array([0.0, 0.1, 0.2], dtype=object),  # A3
        )
        r = validate_run_state(s)
        assert not r.ok
        invs = {e.invariant for e in r.errors}
        assert "A3" in invs and "B2" in invs
        assert len(r.errors) >= 2

    # --- raise_if_invalid ----------------------------------------------

    def test_raise_if_invalid_raises_with_errors(self):
        s = _mutate(_make_state(eval_count=5), best_loss=0.05)
        r = validate_run_state(s)
        with pytest.raises(RunStateValidationException) as exc_info:
            r.raise_if_invalid()
        assert len(exc_info.value.errors) >= 1
        assert any(e.invariant == "B2" for e in exc_info.value.errors)

    def test_raise_if_invalid_noop_when_ok(self):
        validate_run_state(_make_state()).raise_if_invalid()  # must not raise


# ---------------------------------------------------------------------
# CheckpointManager validation gates (#2)
# ---------------------------------------------------------------------


class TestCheckpointManagerValidation:
    def _manager(self, tmp_path, **kw) -> CheckpointManager:
        return CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(),
            **kw,
        )

    def test_save_rejects_invalid_state(self, tmp_path):
        manager = self._manager(tmp_path)
        s = _mutate(_make_state(eval_count=5), best_loss=0.05)  # B2 violation
        with pytest.raises(RunStateValidationException, match="B2"):
            manager.save(s)
        # Nothing was written.
        assert not list(tmp_path.rglob("*.npz"))

    def test_load_rejects_tampered_artifact(self, tmp_path):
        """Save a valid state with validation off, mutate the bytes,
        reload with validation on -> must reject."""
        manager = self._manager(tmp_path, validate_on_save=False)
        valid = _make_state()
        path = manager.save(valid)

        # Tamper: load bytes, break B2, write back without going through
        # the manager. We do this by re-serializing a bad state directly.
        bad = _mutate(valid, best_loss=0.05)
        tampered_bytes = NpzCheckpointSerializer().serialize(bad)
        LocalFilesystemBackend(root=tmp_path).save_bytes(path, tampered_bytes)

        with pytest.raises(RunStateValidationException, match="B2"):
            manager.load(path)

    def test_load_succeeds_when_validation_disabled(self, tmp_path):
        manager = self._manager(
            tmp_path, validate_on_save=False, validate_on_load=False
        )
        bad = _mutate(_make_state(eval_count=5), best_loss=0.05)
        path = manager.save(bad)
        loaded = manager.load(path)  # no raise
        assert loaded.eval_count == 5

    def test_save_load_round_trip_valid_state(self, tmp_path):
        manager = self._manager(tmp_path)
        s = _make_state()
        path = manager.save(s)
        loaded = manager.load(path)
        assert loaded.eval_count == s.eval_count
        assert manager.last_checkpoint_eval == s.eval_count

    def test_legacy_minimal_npz_loads_with_validation_off(self, tmp_path):
        """The pre-invariant legacy NPZ (no metadata/version, missing keys)
        must still load when the organizer disables load-time validation."""
        import io

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            loss_history=np.array([1.0, 2.0], dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array([], dtype=object),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.array([0.0, 0.1], dtype=object),
            eval_count=np.int64(2),
            best_loss=np.float64(1.0),
            best_params=np.array([], dtype=np.float64),
            improvement_count=np.int64(0),
            evals_since_improvement=np.int64(0),
        )
        path = tmp_path / "legacy.npz"
        LocalFilesystemBackend(root=tmp_path).save_bytes(path, buf.getvalue())

        manager = self._manager(tmp_path, validate_on_load=False)
        loaded = manager.load(path)
        assert loaded.eval_count == 2
