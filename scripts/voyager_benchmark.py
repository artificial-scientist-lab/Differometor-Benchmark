from dfbench import (
    EvoxPSO,
    AdamGD,
    NAAdamGD,
    SAGD,
    BotorchBO,
    VoyagerProblem,
    Benchmark,
    AlgorithmConfig,
)

# Setup
problem = VoyagerProblem()
configs = [
    # AlgorithmConfig(AdamGD(problem), {"learning_rate": 0.1}, "Adam-0.1"),
    # AlgorithmConfig(NAAdamGD(problem), {"noise_anneal_iters": 5000}, "NAAdam"),
    # AlgorithmConfig(SAGD(problem), {}, "SAGD-default"),
    # AlgorithmConfig(SAGD(problem), {"use_double_annealing": True}, "SAGD-double"),
    # AlgorithmConfig(EvoxPSO(problem, batch_size=125), {"pop_size": 250}, "PSO-250"),
    # AlgorithmConfig(EvoxPSO(problem, variant="CSO", batch_size=125), {"pop_size": 500}, "CSO-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="CLPSO", batch_size=125), {"pop_size": 500}, "CLPSO-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="FSPSO", batch_size=125), {"pop_size": 500}, "FSPSO-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="SLPSOGS", batch_size=125), {"pop_size": 500}, "SLPSOGS-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="SLPSOUS", batch_size=125), {"pop_size": 500}, "SLPSOUS-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="DMSPSOEL", batch_size=125), {"pop_size": 500}, "DMSPSOEL-500"),
    AlgorithmConfig(BotorchBO(problem), {"initial_samples": 20, "batch_size": 5}, "BoTorch-20-5"),
]

# Run benchmark
benchmark = Benchmark(
    problem=problem,
    success_loss=0,
    configs=configs,
    n_runs=20,
    wall_time_steps=[15,30,60,120,240,480,960],
    random_seed=42,
)

results = benchmark.run_benchmark(save_csv=True, save_run_data=True)
benchmark.print_summary(results)