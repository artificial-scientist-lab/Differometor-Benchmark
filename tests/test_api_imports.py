"""Section 12: Package-level import smoke tests.

Tests 12.1-12.5: verify that all public API symbols are importable.
"""


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
