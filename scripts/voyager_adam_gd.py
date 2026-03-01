"""Test script for AdamGD optimizer."""

from dfbench.problems import VoyagerProblem
from dfbench.algorithms import AdamGD
from dfbench import Objective

# Optimization workflow with Adam
vp = VoyagerProblem()
obj = Objective(
    vp,
    unbounded=True,
    max_time=300,
    verbose=1,
    max_evals=20000,
)

optimizer = AdamGD()


# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    learning_rate=0.1,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
