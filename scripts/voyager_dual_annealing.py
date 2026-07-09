"""Benchmark run: Dual Annealing on VoyagerProblem (single run, ~2 h budget)."""

from dfbench.problems import VoyagerProblem
from dfbench.algorithms import DualAnnealing
from dfbench.core.objective import Objective

SEED = 42
MAX_TIME = 6900  # seconds, leaves ~5 min for saving before the 2 h wall-time
MAX_EVALS = 50_000  # global method; fewer evals are more meaningful

problem = VoyagerProblem()

obj = Objective(
    problem,
    max_time=MAX_TIME,
    max_evals=MAX_EVALS,
    verbose=1,
    print_every=100,
    save_params_history=True,
)

algo = DualAnnealing()
algo.optimize(obj, random_seed=SEED, maxiter=1000, local_refinement=True)

print(f"\nBest loss:   {obj.best_loss:.6f}")
print(f"Evaluations: {obj.eval_count}")
print(f"Time:        {obj.time_elapsed:.1f} s")

obj.output_to_files(hyper_param_str="refine")
obj.save_run_data(algo.algorithm_str, hyper_param_str="refine")
