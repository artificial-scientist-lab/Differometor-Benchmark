"""Section 12: Package-level import smoke tests.

Tests 12.1-12.5: verify that all public API symbols are importable.
Lazy-import regressions use fresh interpreters to avoid cached modules.
"""


import subprocess
import sys
import textwrap


# 12.1
class TestCoreImports:
    def test_import_objective(self):
        from dfbench import Objective  # noqa: F401

    # 12.2
    def test_import_protocols(self):
        from dfbench import ContinuousProblem, OptimizationAlgorithm, AlgorithmType  # noqa: F401

    # 12.3
    def test_import_algorithms(self):
        from dfbench.algorithms import (
            AdamGD,  # noqa: F401
            BFGS,  # noqa: F401
            COBYLA,  # noqa: F401
            COBYQA,  # noqa: F401
            SAGD,  # noqa: F401
            Dogleg,  # noqa: F401
            NAAdamGD,  # noqa: F401
            LBFGSGD,  # noqa: F401
            LBFGSB,  # noqa: F401
            NewtonCG,  # noqa: F401
            NonlinearCG,  # noqa: F401
            SLSQP,  # noqa: F401
            SR1,  # noqa: F401
            TNC,  # noqa: F401
            TrustConstr,  # noqa: F401
            TrustKrylov,  # noqa: F401
            TrustNCG,  # noqa: F401
            EvoxES,  # noqa: F401
            EvoxPSO,  # noqa: F401
            RandomSearch,  # noqa: F401
            BotorchBO,  # noqa: F401
            BotorchTuRBO,  # noqa: F401
            ReSTIR,  # noqa: F401
            VAESampling,  # noqa: F401
        )

    # 12.4
    def test_import_problems(self):
        from dfbench.problems import (
            VoyagerProblem,  # noqa: F401
            VoyagerTuningProblem,  # noqa: F401
            ConstrainedVoyagerProblem,  # noqa: F401
            UIFOProblem,  # noqa: F401
        )

    def test_import_random_uifo_alias(self):
        """RandomUIFOProblem backwards-compat alias is importable."""
        from dfbench.problems import RandomUIFOProblem  # noqa: F401

    # 12.5
    def test_import_benchmark(self):
        from dfbench.benchmark import Benchmark, AlgorithmConfig  # noqa: F401

    # 12.6: submitter/organizer namespace split
    def test_storage_symbols_not_in_top_level(self):
        """Organizer-only storage symbols must not leak into the submitter
        namespace ``dfbench.<name>``. They remain importable from
        ``dfbench.core.storage``; this test guards against re-adding them
        to ``dfbench.__all__`` or re-exporting them from
        ``dfbench/__init__.py``."""
        import dfbench

        organizer_only = [
            "CheckpointManager",
            "CheckpointSerializer",
            "JsonCheckpointSerializer",
            "LocalFilesystemBackend",
            "NpzCheckpointSerializer",
            "NpzRunCollectionSerializer",
            "RunCollectionSerializer",
            "RunDataExporter",
            "RunMetadata",
            "RunPathResolver",
            "RunState",
            "SaveConfig",
            "StorageBackend",
            "validate_run_state",
            "RunStateValidationException",
            "ValidationReport",
            "build_problem_from_spec",
            "register_problem",
            "validate_spec_round_trip",
        ]
        for name in organizer_only:
            assert not hasattr(dfbench, name), (
                f"'{name}' is organizer-only and must not be re-exported at the "
                "top-level dfbench namespace (submitter surface). Import it from "
                "dfbench.core.storage or dfbench.core.problem instead."
            )
            assert name not in dfbench.__all__, (
                f"'{name}' must not appear in dfbench.__all__."
            )

    def test_submitter_symbols_present(self):
        """The submitter-facing symbols stay in the top-level namespace."""
        import dfbench

        for name in (
            "Objective",
            "ContinuousProblem",
            "OptimizationAlgorithm",
            "AlgorithmType",
            "create_parser",
            "t2j",
            "j2t",
        ):
            assert hasattr(dfbench, name), f"'{name}' missing from dfbench"
            assert name in dfbench.__all__, f"'{name}' missing from dfbench.__all__"


def run_isolated(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_imports_do_not_load_implementations():
    run_isolated('''
        import importlib
        import importlib.abc
        import pathlib
        import sys
        import dfbench

        root = pathlib.Path(dfbench.__file__).parent / "algorithms"
        packages = {
            "dfbench." + ".".join(path.parent.relative_to(root.parent).parts)
            for path in root.rglob("__init__.py")
        }

        class BlockImplementations(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("dfbench.algorithms.") and fullname not in packages:
                    raise AssertionError(f"Eager implementation import: {fullname}")

        sys.meta_path.insert(0, BlockImplementations())
        for name in sorted(packages):
            package = importlib.import_module(name)
            assert set(getattr(package, "__all__", ())) <= set(dir(package))
            try:
                package.nonexistent_algorithm
            except AttributeError:
                pass
            else:
                raise AssertionError("Unknown attributes must raise AttributeError")
    ''')


def test_shorthands_work_without_unrelated_optional_dependencies():
    run_isolated('''
        import importlib.abc
        import sys

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {
                    "torch", "evox", "nevergrad", "botorch",
                    "gpytorch", "ax", "hebo", "smac", "cma", "cmaes", "evosax",
                }:
                    raise ModuleNotFoundError(f"Blocked optional dependency: {fullname}", name=fullname)

        sys.meta_path.insert(0, BlockOptional())
        from dfbench.algorithms import AdamGD, JAXOnePlusOneES
        from dfbench.algorithms.gradient_based import AdamGD as GradientAdam
        from dfbench.algorithms.gradient_based.adam_gd import AdamGD as DirectAdam
        from dfbench.algorithms.evolutionary import JAXOnePlusOneES as EvolutionES
        from dfbench.algorithms.gradient_based.scipy import BFGS
        from dfbench.algorithms import BFGS as TopBFGS
        assert AdamGD is GradientAdam is DirectAdam
        assert JAXOnePlusOneES is EvolutionES
        assert BFGS is TopBFGS

        import dfbench.algorithms as algorithms
        assert algorithms.__dict__["AdamGD"] is AdamGD
        for package in (algorithms, sys.modules["dfbench.algorithms.evolutionary"]):
            try:
                package.EvoxES
            except ImportError as exc:
                assert "evox and torch are required" in str(exc)
            else:
                raise AssertionError("Requesting EvoxES must report missing dependencies")
            assert "EvoxES" not in package.__dict__
        assert algorithms.AdamGD is DirectAdam
    ''')
