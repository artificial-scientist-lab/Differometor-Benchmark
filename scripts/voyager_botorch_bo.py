"""Test script for BotorchBO optimizer."""

from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import BotorchBO

# Optimization workflow with Bayesian Optimization
vp = ConstrainedVoyagerProblem()

optimizer = BotorchBO(vp)

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    max_iterations=200,
    verbose=1,
    save_run_to_file=True,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
