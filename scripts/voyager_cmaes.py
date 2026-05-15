"""Test script for EvoxES (CMA-ES variant) optimizer."""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import EvoxES

problem = VoyagerProblem()

obj = Objective(
    problem,
    max_evals=200_000,
    verbose=1,
    print_every=1000,
    save_params_history=True,
)

optimizer = EvoxES(batch_size=50, variant="CMAES")

optimizer.optimize(
    objective=obj,
    max_iterations=2000,
    pop_size=100,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
