"""Test script for BotorchTuRBO optimizer."""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import BotorchTuRBO

problem = VoyagerProblem()

obj = Objective(
    problem,
    max_evals=5_000,
    verbose=1,
    print_every=20,
    save_params_history=True,
)

optimizer = BotorchTuRBO(batch_size=4)

optimizer.optimize(
    problem_objective=obj,
    max_iterations=200,
    n_initial=None,  # defaults to 2 * dim
    acquisition_batch_size=4,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
