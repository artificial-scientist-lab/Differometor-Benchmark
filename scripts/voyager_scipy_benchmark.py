"""Benchmark a batch of SciPy-based optimizers on Voyager."""

from dfbench.algorithms import (
    BFGS,
    COBYLA,
    COBYQA,
    Dogleg,
    LBFGSB,
    NewtonCG,
    NonlinearCG,
    SLSQP,
    SR1,
    TNC,
    TrustConstr,
    TrustKrylov,
    TrustNCG,
)
from dfbench.benchmark import AlgorithmConfig, Benchmark
from dfbench.problems import VoyagerProblem


problem = VoyagerProblem()

configs = [
    AlgorithmConfig(BFGS(), {"gtol": 1e-5}, name="SciPy-BFGS"),
    AlgorithmConfig(LBFGSB(), {"gtol": 1e-5, "maxcor": 10}, name="SciPy-LBFGSB"),
    AlgorithmConfig(NonlinearCG(), {"gtol": 1e-5}, name="SciPy-NonlinearCG"),
    AlgorithmConfig(NewtonCG(), {"xtol": 1e-5}, name="SciPy-NewtonCG"),
    AlgorithmConfig(TrustNCG(), {"gtol": 1e-5}, name="SciPy-TrustNCG"),
    AlgorithmConfig(TrustKrylov(), {"gtol": 1e-5}, name="SciPy-TrustKrylov"),
    AlgorithmConfig(
        TrustConstr(),
        {"gtol": 1e-6, "initial_tr_radius": 1.0},
        name="SciPy-TrustConstr",
    ),
    AlgorithmConfig(TNC(), {"maxfun": 200}, name="SciPy-TNC"),
    AlgorithmConfig(SLSQP(), {"ftol": 1e-6, "maxiter": 200}, name="SciPy-SLSQP"),
    AlgorithmConfig(COBYQA(), {"maxfev": 200}, name="SciPy-COBYQA"),
    AlgorithmConfig(COBYLA(), {"maxiter": 200, "rhobeg": 1.0}, name="SciPy-COBYLA"),
    AlgorithmConfig(Dogleg(), {"gtol": 1e-5}, name="SciPy-Dogleg"),
    AlgorithmConfig(SR1(), {"gtol": 1e-6}, name="SciPy-SR1"),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=50,
    configs=configs,
    n_runs=3,
    max_time=90,
    n_time_samples=9,
    random_seed=42,
)

results = benchmark.run(save_csv=False)
benchmark.print_summary(results)
