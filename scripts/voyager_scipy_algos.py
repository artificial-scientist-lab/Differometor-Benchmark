"""Run every new SciPy algorithm for 3 minutes on the Voyager Problem."""

from dfbench.algorithms import (
    COBYLA,
    COBYQA,
    Dogleg,
    LBFGSB,
    SR1,
)
from dfbench import Objective
from dfbench.problems import VoyagerProblem

ALGORITHMS = [
    # BFGS(),
    LBFGSB(),
    # NonlinearCG(),
    # NewtonCG(),
    # TrustNCG(),
    # TrustKrylov(),
    # TrustConstr(),
    # TNC(),
    # SLSQP(),
    COBYQA(),
    COBYLA(),
    Dogleg(),
    SR1(),
]

for algo in ALGORITHMS:
    vp = VoyagerProblem()
    obj = Objective(vp, max_time=60, verbose=1, print_every=2)
    algo.optimize(obj, random_seed=42)
    print(
        f"\n[{algo.algorithm_str}] best loss: {obj.best_loss:.6f}  evals: {obj.eval_count}\n"
    )
