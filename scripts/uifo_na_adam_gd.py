"""Test script for NAAdamGD optimizer."""

import argparse

from dfbench.problems import UIFOProblem, VoyagerProblem
from dfbench.algorithms import NAAdamGD
from dfbench import Objective

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=None)
seed = parser.parse_args().seed

# Optimization workflow with NA-Adam
problem = UIFOProblem(topology_seed=seed)
obj = Objective(
    problem,
    verbose=1,
    max_time=60*60*24,
    print_every=1000,
    save_params_history=True,
    save_to_file_every=1000,
    display_mode="log",
)

optimizer = NAAdamGD()

# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    learning_rate=0.1,
    patience=None,
    random_seed=seed,
    noise_schedule="linear",
    noise_anneal_budget_fraction=0.5,
)

obj.save_run_data()

print("Best loss:")
print(f"    {obj.best_loss:.6f}")
print("Total evaluations:")
print(f"    {obj.eval_count}")
print("First parameters:")
print(f"    {obj.params_history_bounded[0]}")
print("Best parameters:")
print(f"    {obj.best_params_bounded}")
print("Seed:")
print(f"    {seed}")
