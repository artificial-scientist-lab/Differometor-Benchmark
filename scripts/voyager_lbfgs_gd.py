"""Test script for LBFGSGD optimizer."""

from dfbench.algorithms import LBFGSGD
from dfbench.core.objective import Objective
from dfbench.problems import VoyagerProblem

# Create problem
problem = VoyagerProblem()

# Create Objective wrapper with configuration
obj = Objective(
    problem,
    max_time=60,
    verbose=1,
    print_every=5,
    save_params_history=True,
)

# Create optimizer
optimizer = LBFGSGD()

# Run optimization
optimizer.optimize(
    obj,
    random_seed=420,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

# Optionally save results and create plots
obj.output_to_files(hyper_param_str="standard")
obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
