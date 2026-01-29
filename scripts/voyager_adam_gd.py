"""Test script for AdamGD optimizer."""

from dfbench.problems import VoyagerProblem
from dfbench.algorithms import AdamGD

# Optimization workflow with Adam
vp = VoyagerProblem()
optimizer = AdamGD(vp)

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    max_iterations=2000,
    verbose=1,
    plot_loss=True,
    save_run_to_file=True,
    print_every=100,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
