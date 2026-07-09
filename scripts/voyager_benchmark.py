from dfbench.problems import VoyagerProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.algorithms import (
    AdamGD,
    SAGD,
    NAAdamGD,
    RandomSearch,
    EvoxES,
    EvoxPSO,
    BotorchBO,
    BotorchTuRBO,
    VAESampling,
)

problem = VoyagerProblem()

configs = [
    # Evolutionary
    AlgorithmConfig(
        EvoxES(batch_size=5, variant="CMAES"), {"pop_size": 50}, name="CMA-ES"
    ),
    AlgorithmConfig(EvoxPSO(batch_size=5, variant="PSO"), {"pop_size": 50}, name="PSO"),
    AlgorithmConfig(RandomSearch(batch_size=50), {}, name="RandomSearch"),
    # Gradient-based
    AlgorithmConfig(AdamGD(), {"learning_rate": 0.1, "patience": 500}, name="Adam-GD"),
    AlgorithmConfig(
        SAGD(), {"learning_rate": 0.1, "patience": 500, "T0": 15.0}, name="SA-GD"
    ),
    AlgorithmConfig(
        NAAdamGD(),
        {"learning_rate": 0.1, "patience": 500, "noise_std_start": 0.3},
        name="NA-Adam",
    ),
    # Surrogate-based
    AlgorithmConfig(
        BotorchBO(batch_size=3),
        {"n_initial": 10, "acquisition_batch_size": 3, "max_iterations": 20},
        name="BoTorch-BO",
    ),
    AlgorithmConfig(
        BotorchTuRBO(batch_size=3),
        {"n_initial": 10, "acquisition_batch_size": 3, "max_iterations": 20},
        name="TuRBO",
    ),
    # Generative
    AlgorithmConfig(
        VAESampling(batch_size=3),
        {
            "n_initial": 10,
            "vae_training_samples": 100,
            "vae_epochs": 50,
        },
        name="VAE-Sampling",
    ),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=50,
    configs=configs,
    n_runs=3,
    max_time=90,
    n_time_samples=9,
)

results = benchmark.run(save_csv=False)
benchmark.print_summary(results)
