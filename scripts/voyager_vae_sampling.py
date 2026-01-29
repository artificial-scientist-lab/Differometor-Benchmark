"""Test script for VAESampling optimizer."""

from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import VAESampling

# Optimization workflow with VAE Sampling
vp = ConstrainedVoyagerProblem()

optimizer = VAESampling(vp, batch_size=64, hidden_dim=256, num_blocks=4, use_objective_guidance=True)

# Run optimization - returns Objective instance
obj = optimizer.optimize(
    max_iterations=2000,
    sampling_time_percentage=0.5,
    verbose=1,
    save_run_to_file=True,
    print_every=100,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
