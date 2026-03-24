"""Benchmark script for the CMA-family batch.

Runs all nine CMA-family algorithms on the VoyagerProblem and saves
per-run result files.  Intended to be submitted via srun:

    srun -p a100-galvani --gres=gpu:1 --time=0-02:00 --pty python scripts/voyager_cma_family.py

Algorithms covered:
  - PyCMACMAES        (pycma, vanilla)
  - PyCMAActiveCMAES  (pycma, active CMA)
  - PyCMAIPOP         (pycma, IPOP restarts)
  - PyCMABIPOP        (pycma, BIPOP restarts)
  - CMAESSepCMA       (cmaes, diagonal covariance)
  - EvosaxMAES        (evosax, matrix adaptation)
  - EvosaxLMMAES      (evosax, limited-memory MA)
  - JAXOnePlusOneES   (native JAX, (1+1)-ES)
  - JAXMuLambdaES     (native JAX, (mu,lambda)-ES)

Edit ``GLOBAL_MAX_EVALS``, ``POP_SIZE``, and ``RANDOM_SEED`` to taste.
"""

from __future__ import annotations

MAX_EVALS = 100_000
POP_SIZE = 50
RANDOM_SEED = 42

from dfbench import Objective
from dfbench.problems import VoyagerProblem

problem = VoyagerProblem()

# ---------------------------------------------------------------------------
# Algorithm configs: (class, kwargs_for_init, kwargs_for_optimize)
# ---------------------------------------------------------------------------

configs: list[tuple] = []

# --- pycma ---
try:
    from dfbench.algorithms.evolutionary.pycma_cmaes import (
        PyCMACMAES,
        PyCMAActiveCMAES,
        PyCMAIPOP,
        PyCMABIPOP,
    )

    configs += [
        (PyCMACMAES(batch_size=POP_SIZE), {"pop_size": POP_SIZE}),
        (PyCMAActiveCMAES(batch_size=POP_SIZE), {"pop_size": POP_SIZE}),
        (PyCMAIPOP(batch_size=POP_SIZE), {"pop_size": POP_SIZE, "max_restarts": 5}),
        (PyCMABIPOP(batch_size=POP_SIZE), {"pop_size": POP_SIZE, "max_restarts": 10}),
    ]
except ImportError:
    print("pycma not installed — skipping pycma algorithms.")

# --- cmaes ---
try:
    from dfbench.algorithms.evolutionary.cmaes_sep_cma import CMAESSepCMA

    configs.append((CMAESSepCMA(batch_size=POP_SIZE), {"pop_size": POP_SIZE}))
except ImportError:
    print("cmaes not installed — skipping CMAESSepCMA.")

# --- evosax ---
try:
    from dfbench.algorithms.evolutionary.evosax_es import EvosaxMAES, EvosaxLMMAES

    configs += [
        (EvosaxMAES(batch_size=POP_SIZE), {"pop_size": POP_SIZE}),
        (EvosaxLMMAES(batch_size=POP_SIZE), {"pop_size": POP_SIZE}),
    ]
except ImportError:
    print("evosax not installed — skipping evosax algorithms.")

# --- native JAX ---
from dfbench.algorithms.evolutionary.jax_es import JAXOnePlusOneES, JAXMuLambdaES

configs += [
    (JAXOnePlusOneES(), {}),
    (JAXMuLambdaES(batch_size=POP_SIZE), {"mu": POP_SIZE // 5, "lam": POP_SIZE}),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

for algo, opt_kwargs in configs:
    print(f"\n{'='*60}")
    print(f"Running {algo.algorithm_str} ...")

    obj = Objective(
        problem,
        max_evals=MAX_EVALS,
        verbose=1,
        print_every=5000,
        save_params_history=True,
    )

    algo.optimize(
        problem_objective=obj,
        random_seed=RANDOM_SEED,
        **opt_kwargs,
    )

    print(f"  Best loss:   {obj.best_loss:.6f}")
    print(f"  Evaluations: {obj.eval_count}")

    obj.save_run_data(algo.algorithm_str, hyper_param_str=f"pop{POP_SIZE}")

print("\nAll CMA-family runs complete.")
