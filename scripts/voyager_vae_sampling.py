"""Test script for VAESampling optimizer."""

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import VAESampling

problem = ConstrainedVoyagerProblem()

obj = Objective(
    problem,
    max_time=120,
    verbose=1,
    print_every=1,
)

optimizer = VAESampling(batch_size_sampling=64, batch_size_bo=1)

optimizer.optimize(
    objective=obj,
    max_iterations=None,
    vae_training_samples=None,
    sampling_budget_fraction=0.25,
    vae_epochs=100,
    vae_train_batch_size=64,
    hidden_dim=256,
    num_blocks=4,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
