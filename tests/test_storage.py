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
        # A minimal NPZ without metadata/version — must still load.
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
        r = RunPathResolver(root="./data/run_data")
        p = r.checkpoint_path(
            problem_name="voyager",
            algorithm_name="adam/gd",
            timestamp="2026-01-01_00-00-00",
            hyper_param_str="lr0.1",
            max_time=100.0,
            max_evals=1000,
        )
        assert p.name == "voyager_adam_gd_2026-01-01_00-00-00.npz"
        assert "time100s_evals1000" in p.parts
        assert "lr0.1" in p.parts

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
            resolver=RunPathResolver(root=str(tmp_path)),
        )
        state = _make_state()
        path = manager.save(state)
        assert path.exists()
        assert manager.last_checkpoint_eval == state.eval_count

        loaded = manager.load(path)
        assert loaded.eval_count == state.eval_count
        # Cached path means a second save without overrides rewrites same file
        path2 = manager.save(state)
        assert path2 == path

    def test_maybe_save_skips_when_not_due(self, tmp_path):
        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(root=str(tmp_path)),
        )
        called = {"n": 0}

        def factory():
            called["n"] += 1
            return _make_state()

        # eval_count=3, save_every=5 -> not due
        assert manager.maybe_save(factory, 3, 5) is None
        assert called["n"] == 0
        # eval_count=5 -> due
        assert manager.maybe_save(factory, 5, 5) is not None
        assert called["n"] == 1

    def test_explicit_path_overrides_resolver(self, tmp_path):
        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(root="./data/ignored"),
        )
        state = _make_state()
        explicit = tmp_path / "custom.npz"
        path = manager.save(state, explicit_path=explicit)
        assert path == explicit
        assert explicit.exists()


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
            resolver=RunPathResolver(root=str(tmp_path)),
        )
        assert manager.resolver.extension == "json"
        state = _make_state()
        path = manager.save(state)
        assert path.suffix == ".json"
