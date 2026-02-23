"""Test script for BotorchBO optimizer."""

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import BotorchBO

problem = ConstrainedVoyagerProblem()

obj = Objective(
    problem,
    max_evals=2_000,
    verbose=1,
    print_every=10,
    save_params_history=True,
)

optimizer = BotorchBO()

optimizer.optimize(
    problem_objective=obj,
    max_iterations=200,
    n_initial=10,
    batch_size=1,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
