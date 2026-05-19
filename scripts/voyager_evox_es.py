"""Test script for EvoxES (SNES variant) optimizer."""

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import EvoxES

problem = ConstrainedVoyagerProblem()

obj = Objective(
    problem,
    max_evals=1_000_000,
    verbose=1,
    print_every=5000,
    save_params_history=True,
)

optimizer = EvoxES(batch_size=125, variant="SNES")

optimizer.optimize(
    objective=obj,
    max_iterations=2000,
    pop_size=500,
    random_seed=42,
)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")

obj.save_run_data(optimizer.algorithm_str, hyper_param_str="standard")
