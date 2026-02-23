"""Test script for ReSTIR optimizer."""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import ReSTIR

# Create problem and objective
problem = VoyagerProblem()
obj = Objective(
    problem,
    max_time=3000,
    verbose=1,
    print_every=100,
    save_params_history=True,
)

# Create optimizer
optimizer = ReSTIR(batch_size=10)

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    problem_objective=obj,
    random_seed=42,
    n_total_samples=1_000_000,
    n_initial_reference_samples=10_000,
    reservoir_size=1000,
    n_gd_candidates=100,
    k_neighbors=10,
    temperature=1.0,
    gd_steps=1000,
    gd_learning_rate=0.1,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
