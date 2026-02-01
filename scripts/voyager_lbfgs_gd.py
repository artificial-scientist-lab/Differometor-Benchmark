"""Test script for LBFGSGD optimizer."""

from dfbench.algorithms import LBFGSGD
from dfbench.problems import VoyagerProblem

# Optimization workflow with LBFGS
vp = VoyagerProblem()
optimizer = LBFGSGD(vp)

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    random_seed=42,
    max_iterations=2000,
    verbose=1,
    plot_loss=True,
    save_run_to_file=True,
    print_every=100,
)

print(f"Best loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
