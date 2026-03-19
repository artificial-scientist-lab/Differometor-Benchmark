"""Run all four Powell-style DFO algorithms for 3 minutes each on the Voyager Problem."""

from dfbench.algorithms import PDFOUOBYQA, PDFONEWUOA, PDFOLINCOA, PyBOBYQA
from dfbench import Objective
from dfbench.problems import VoyagerProblem

ALGORITHMS = [
    PDFOUOBYQA(n_restarts=3),
    PDFONEWUOA(n_restarts=3),
    PDFOLINCOA(n_restarts=3),
    PyBOBYQA(seek_global_minimum=True, n_restarts=2),
]

for algo in ALGORITHMS:
    vp = VoyagerProblem()
    obj = Objective(vp, max_time=180, verbose=1, print_every=5)
    algo.optimize(obj, random_seed=42)
    print(f"\nBest loss: {obj.best_loss:.6f}")
    print(f"Total evaluations: {obj.eval_count}\n")
