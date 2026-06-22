"""Test script for VeLO optimizer."""

from dfbench.problems import VoyagerProblem
from dfbench.algorithms import VeLO
from dfbench import Objective

# Optimization workflow with VeLO
vp = VoyagerProblem()
obj = Objective(
    vp,
    verbose=1,
    max_evals=2000,
    print_every=10,
    save_params_history=True,
)

optimizer = VeLO()


# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
print(f"First parameters: {obj.params_history_bounded[0]}")
print(f"Last parameters: {obj.params_history_bounded[-1]}")
