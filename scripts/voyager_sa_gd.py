"""Test script for SAGD optimizer."""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import SAGD

problem = VoyagerProblem()

obj = Objective(
    problem,
    unbounded=True,
    max_evals=20_000,
    verbose=1,
    print_every=100,
    save_params_history=True,
)

optimizer = SAGD()

optimizer.optimize(
    problem_objective=obj,
    learning_rate=0.1,
    patience=1000,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
