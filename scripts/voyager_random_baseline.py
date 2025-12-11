from dfbench.algorithms import RandomSearch
from dfbench.problems.voyager_problem import VoyagerProblem

problem = VoyagerProblem(use_sigmoid_bounding=False)
algorithm = RandomSearch(problem, batch_size=125)

# Estimate baseline statistics over multiple runs
stats = algorithm.estimate_baseline_statistics(
    n_samples=1000,
    n_runs=20,
    seed_start=40,
)

print(f"\nRandom baseline over 20 runs (1000 samples each):")
print(f"Mean: {stats['mean']:.6f}")
print(f"Std: {stats['std']:.6f}")
print(f"Min: {stats['min']:.6f}")
print(f"Max: {stats['max']:.6f}")
print(f"Median: {stats['median']:.6f}")
