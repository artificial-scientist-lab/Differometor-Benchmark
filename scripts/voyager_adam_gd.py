"""Test script for AdamGD optimizer."""

from dfbench.problems import VoyagerProblem, RandomUIFOProblem
from dfbench.algorithms import AdamGD
from dfbench import Objective

# Optimization workflow with Adam
vp = VoyagerProblem()
obj = Objective(
    vp,
    verbose=1,
    max_evals=500,
    print_every=5,
    save_params_history=True,
)

optimizer = AdamGD()


# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    random_seed=42,
    learning_rate=0.1,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
print(f"First parameters: {obj.params_history_bounded[0]}")
print(f"Last parameters: {obj.params_history_bounded[-1]}")
