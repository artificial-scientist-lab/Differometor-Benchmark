"""Benchmark script for MADS / OrthoMADS direct-search algorithms.

Runs both OmadsMADS (search + poll) and OmadsOrthoMADS (poll only) on
the Voyager problem. These are derivative-free local explorers suited
for rugged landscapes with moderate budgets.

Usage (via slurm for compute-heavy problems):
    srun -p a100-galvani --gres=gpu:1 --time=0-00:50 --pty bash -I
    source .venv/bin/activate
    python scripts/voyager_omads.py
"""

from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import OmadsMADS, OmadsOrthoMADS

problem = VoyagerProblem()

# --- MADS (search + poll) ---------------------------------------------------

obj_mads = Objective(
    problem,
    max_evals=5_000,
    verbose=1,
    print_every=500,
    save_params_history=True,
)

optimizer_mads = OmadsMADS(psize_init=1.0, tol=1e-9, ns=4)

optimizer_mads.optimize(
    objective=obj_mads,
    random_seed=42,
)

print(f"\n[MADS] Best loss: {obj_mads.best_loss:.6f}")
print(f"[MADS] Total evaluations: {obj_mads.eval_count}")

obj_mads.save_run_data(optimizer_mads.algorithm_str, hyper_param_str="standard")

# --- OrthoMADS (poll only) ---------------------------------------------------

obj_ortho = Objective(
    problem,
    max_evals=5_000,
    verbose=1,
    print_every=500,
    save_params_history=True,
)

optimizer_ortho = OmadsOrthoMADS(psize_init=1.0, tol=1e-9)

optimizer_ortho.optimize(
    objective=obj_ortho,
    random_seed=42,
)

print(f"\n[OrthoMADS] Best loss: {obj_ortho.best_loss:.6f}")
print(f"[OrthoMADS] Total evaluations: {obj_ortho.eval_count}")

obj_ortho.save_run_data(optimizer_ortho.algorithm_str, hyper_param_str="standard")
