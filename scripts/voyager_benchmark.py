from dfbench import (
    EvoxPSO,
    AdamGD,
    NAAdamGD,
    SAGD,
    BotorchBO,
    VoyagerProblem,
    Benchmark,
    AlgorithmConfig,
    RandomUIFOProblem,
    ConstrainedVoyagerProblem,
)

# Setup
problem = ConstrainedVoyagerProblem()
configs = [
    # AlgorithmConfig(AdamGD(problem), {"learning_rate": 0.1}, "Adam-0.1"),
    # AlgorithmConfig(NAAdamGD(problem), {"noise_anneal_iters": 5000, "noise_schedule": "linear"}, "NAAdam-lin"),
    # AlgorithmConfig(NAAdamGD(problem), {"noise_anneal_iters": 5000, "noise_schedule": "exponential"}, "NAAdam-exp"),
    # AlgorithmConfig(SAGD(problem), {}, "SAGD-default"),
    # AlgorithmConfig(SAGD(problem), {"use_double_annealing": True}, "SAGD-double"),
    # AlgorithmConfig(EvoxPSO(problem, batch_size=125), {"pop_size": 250}, "PSO-250"),
    # AlgorithmConfig(EvoxPSO(problem, variant="CSO", batch_size=125), {"pop_size": 500}, "CSO-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="CLPSO", batch_size=125), {"pop_size": 500}, "CLPSO-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="FSPSO", batch_size=125), {"pop_size": 500}, "FSPSO-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="SLPSOGS", batch_size=125), {"pop_size": 500}, "SLPSOGS-500"),
    # AlgorithmConfig(EvoxPSO(problem, variant="SLPSOUS", batch_size=125), {"pop_size": 500}, "SLPSOUS-500"),
    # AlgorithmConfig(EvoxES(problem, batch_size=50, variant="SNES"), {"pop_size": 50}, "SNES-50"),
    # AlgorithmConfig(BotorchBO(problem), {}, "BotorchBO"),
    # AlgorithmConfig(BotorchTuRBO(problem), {}, "BotorchTuRBO"),
]

# Run benchmark
benchmark = Benchmark(
    problem=problem,
    success_loss=0,
    configs=configs,
    n_runs=20,
    wall_time_steps=[15,30,60,120,240,480,720,960,1200],
    random_seed=43,
)

results = benchmark.run_benchmark(save_csv=True, save_run_data=True)
benchmark.print_summary(results)
