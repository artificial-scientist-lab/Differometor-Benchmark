"""Test script for EvoxES optimizer."""

from dfbench.problems import VoyagerProblem
from dfbench.algorithms import EvoxES

vp = VoyagerProblem()

optimizer = EvoxES(problem=vp, batch_size=50, variant="CMAES")

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    max_iterations=2000,
    pop_size=100,
    verbose=1,
    save_run_to_file=True,
    print_every=100,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
