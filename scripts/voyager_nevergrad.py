"""Benchmark script for the Nevergrad batch (rugged-landscape controls).

Runs OnePlusOne, TBPSA, and NGOpt on the Voyager problem and saves results.
Intended for comparing lightweight derivative-free baselines.
"""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import NevergradOnePlusOne, NevergradTBPSA, NevergradNGOpt

problem = VoyagerProblem()

SHARED_BUDGET = 50_000
SEED = 42

configs = [
    ("ng_oneplusone", NevergradOnePlusOne(), {"n_restarts": 3}),
    ("ng_tbpsa", NevergradTBPSA(), {"n_restarts": 1, "num_evaluations": 1}),
    ("ng_ngopt", NevergradNGOpt(), {"n_restarts": 1}),
]

for name, optimizer, extra_kwargs in configs:
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")

    obj = Objective(
        problem,
        max_evals=SHARED_BUDGET,
        verbose=1,
        print_every=5000,
        save_params_history=True,
    )

    optimizer.optimize(
        objective=obj,
        random_seed=SEED,
        **extra_kwargs,
    )

    print(f"\n{name} — Best loss: {obj.best_loss:.6f}  |  Evals: {obj.eval_count}")
    obj.save_run_data(name, hyper_param_str="standard")
