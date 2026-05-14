"""Test script for VAESampling optimizer."""

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import VAESampling

problem = ConstrainedVoyagerProblem()

obj = Objective(
    problem,
    max_evals=10_000,
    verbose=1,
    print_every=100,
    save_params_history=True,
)

optimizer = VAESampling(batch_size=64)

optimizer.optimize(
    problem_objective=obj,
    max_iterations=50,
    vae_training_samples=1000,
    vae_epochs=100,
    vae_train_batch_size=64,
    hidden_dim=256,
    num_blocks=4,
    use_objective_guidance=True,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
