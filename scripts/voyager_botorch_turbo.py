"""Test script for BotorchTuRBO optimizer."""

from dfbench.problems import VoyagerProblem
from dfbench.algorithms import BotorchTuRBO

# Optimization workflow with TuRBO
vp = VoyagerProblem()

optimizer = BotorchTuRBO(vp)

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    max_iterations=200,
    verbose=1,
    save_run_to_file=True,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
