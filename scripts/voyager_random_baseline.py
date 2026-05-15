"""Estimate random search baseline statistics for the ConstrainedVoyagerProblem."""

import numpy as np

from dfbench import Objective
from dfbench.algorithms import RandomSearch
from dfbench.problems import ConstrainedVoyagerProblem

n_runs = 20
n_samples = 1000
seed_start = 40

best_losses = []
for run in range(n_runs):
    problem = ConstrainedVoyagerProblem()
    obj = Objective(problem, max_evals=n_samples)
    algorithm = RandomSearch(batch_size=125)
    algorithm.optimize(objective=obj, random_seed=seed_start + run)
    best_losses.append(float(obj.best_loss))

best_losses = np.array(best_losses)
print(f"\nRandom baseline over {n_runs} runs ({n_samples} samples each):")
print(f"Mean:   {np.mean(best_losses):.6f}")
print(f"Std:    {np.std(best_losses):.6f}")
print(f"Min:    {np.min(best_losses):.6f}")
print(f"Max:    {np.max(best_losses):.6f}")
print(f"Median: {np.median(best_losses):.6f}")
