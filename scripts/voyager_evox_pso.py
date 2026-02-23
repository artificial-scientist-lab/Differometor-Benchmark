"""Test script for EvoxPSO optimizer."""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import EvoxPSO

problem = VoyagerProblem()

obj = Objective(
    problem,
    max_evals=100_000,
    verbose=1,
    print_every=500,
    save_params_history=True,
)

optimizer = EvoxPSO(batch_size=50, variant="PSO")

optimizer.optimize(
    problem_objective=obj,
    max_iterations=2000,
    pop_size=50,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
