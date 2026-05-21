import argparse
import logging
import sys
import warnings


from dfbench.problems import UIFOProblem
from dfbench import Objective

# ── Combined algorithm list (everything merged into main from the
#    seven algorithm/* branches, grouped by family) ────────────────────

from dfbench.algorithms import (
    # Nevergrad baselines
    NevergradOnePlusOne,
    NevergradTBPSA,
    NevergradNGOpt,
    # CMA family (cma-es branch)
    PyCMACMAES,
    PyCMAActiveCMAES,
    PyCMAIPOP,
    PyCMABIPOP,
    CMAESSepCMA,
    JAXOnePlusOneES,
    JAXMuLambdaES,
    # Native-JAX custom/hybrid batch
    ASAMJAX,
    AdamToLBFGSJAX,
    EntropySGDJAX,
    GDRestartsJAX,
    GaussianSmoothingGDJAX,
    NoisyAdamJAX,
    OAdamJAX,
    OGDJAX,
    PerturbedGDJAX,
    SGHMCJAX,
    SGLDJAX,
    # Powell-style trust-region DFO
    PDFOUOBYQA,
    PDFONEWUOA,
    PDFOLINCOA,
    # SciPy classics + global search
    NelderMead,
    Powell,
    BasinHopping,
    DualAnnealing,
    # Bayesian Optimization batch
    BAxUS,
    BotorchqNEI,
    BotorchqKG,
    REMBO,
    GEBO,
    LineBO,
    TuRBOLBFGS,
)

# Silence BoTorch / GPyTorch / Ax / SMAC warning spam.
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
for _name in (
    "py.warnings",
    "botorch",
    "gpytorch",
    "linear_operator",
    "ax",
    "smac",
    "ConfigSpace",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


ALGORITHMS = [
    # Nevergrad baselines
    NevergradOnePlusOne,
    NevergradTBPSA,
    NevergradNGOpt,
    # CMA family
    PyCMACMAES,
    PyCMAActiveCMAES,
    PyCMAIPOP,
    PyCMABIPOP,
    CMAESSepCMA,
    JAXOnePlusOneES,
    JAXMuLambdaES,
    # Native-JAX custom/hybrid batch
    ASAMJAX,
    AdamToLBFGSJAX,
    EntropySGDJAX,
    GDRestartsJAX,
    GaussianSmoothingGDJAX,
    NoisyAdamJAX,
    OAdamJAX,
    OGDJAX,
    PerturbedGDJAX,
    SGHMCJAX,
    SGLDJAX,
    # Powell-style trust-region DFO
    PDFOUOBYQA,
    PDFONEWUOA,
    PDFOLINCOA,
    # SciPy classics + global search
    NelderMead,
    Powell,
    BasinHopping,
    DualAnnealing,
    # Bayesian Optimization batch
    BAxUS,
    BotorchqNEI,
    BotorchqKG,
    REMBO,
    GEBO,
    LineBO,
    TuRBOLBFGS,
]

# ── CLI ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="UIFO sweep over the diverse set of algorithms merged into main "
    "(MADS, Nevergrad, CMA family, JAX custom/hybrid, Powell DFO, "
    "SciPy classics + global search, Bayesian Optimization batch).",
)
parser.add_argument(
    "-a",
    "--algo",
    required=True,
    help=f"Algorithm index (0-{len(ALGORITHMS) - 1}), or 'list' to print the table.",
)
parser.add_argument("-s", "--seed", type=int, default=0, help="Run seed (0-24).")
args = parser.parse_args()

if args.algo == "list":
    print(f"{'Index':<6} {'Class':<30} {'algorithm_str'}")
    print("-" * 60)
    for i, cls in enumerate(ALGORITHMS):
        print(f"{i:<6} {cls.__name__:<30} {cls.algorithm_str}")
    sys.exit(0)

algo_idx = int(args.algo)
if algo_idx < 0 or algo_idx >= len(ALGORITHMS):
    print(f"Error: --algo must be 0-{len(ALGORITHMS) - 1}, got {algo_idx}")
    sys.exit(1)

seed = args.seed
AlgClass = ALGORITHMS[algo_idx]
print(f"Algorithm: {AlgClass.__name__} (index {algo_idx}), seed: {seed}")

# ── Run ───────────────────────────────────────────────────────────────

problem = UIFOProblem(topology_seed=seed)
obj = Objective(
    problem,
    verbose=1,
    max_time=4 * 60 * 60,  # 4 hours
    print_every=1000,
    save_params_history=True,
    save_to_file_every=1000,
    display_mode="log",
)

optimizer = AlgClass()
optimizer.optimize(obj, random_seed=seed)

obj.save_run_data()

print("Best loss:")
print(f"    {obj.best_loss:.6f}")
print("Total evaluations:")
print(f"    {obj.eval_count}")
print("Best parameters:")
print(f"    {obj.best_params_bounded}")
print(f"Algorithm: {AlgClass.algorithm_str}, Seed: {seed}")
