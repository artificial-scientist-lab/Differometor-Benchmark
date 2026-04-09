"""Test script for AdamGD optimizer."""

import argparse

from dfbench.problems import UIFOProblem
from dfbench.algorithms import AdamGD
from dfbench import Objective

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=42)
seed = parser.parse_args().seed

# Optimization workflow with Adam
problem = UIFOProblem(topology_seed=seed)
obj = Objective(
    problem,
    verbose=0,
    max_time= 60*60*24,
    print_every=1000,
    save_params_history=True,
    save_grad_history=False,
    save_to_file_every=1000,
)

optimizer = AdamGD()


# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    learning_rate=0.1,
    patience=None,
)

obj.save_run_data()

print("Best loss:")
print(f"    {obj.best_loss:.6f}")
print("Total evaluations:")
print(f"    {obj.eval_count}")
print("First parameters:")
print(f"    {obj.params_history_bounded[0]}")
print("Last parameters:")
print(f"    {obj.params_history_bounded[-1]}")
