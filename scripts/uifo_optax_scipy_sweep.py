import argparse
import sys

from dfbench.problems import UIFOProblem
from dfbench import Objective

# ── Combined algorithm list (optax first, then scipy) ─────────────────

from dfbench.algorithms import (
    # Optax (32 algorithms)
    OptaxAdam,
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxAdafactor,
    OptaxAMSGrad,
    OptaxAdaGrad,
    OptaxAdaDelta,
    OptaxAdaMax,
    OptaxAdaMaxW,
    OptaxAdan,
    OptaxLion,
    OptaxLAMB,
    OptaxNadam,
    OptaxNadamW,
    OptaxRMSProp,
    OptaxRProp,
    OptaxRAdam,
    OptaxSGD,
    OptaxSGDM,
    OptaxNAG,
    OptaxNoisySGD,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxLookahead,
    OptaxScheduleFreeAdam,
    OptaxYogi,
    OptaxNovoGrad,
    OptaxOGD,
    OptaxOAdam,
    OptaxSignSGD,
    OptaxSignum,
    OptaxSM3,
    OptaxLBFGS,
    # SciPy (13 algorithms)
    BFGS,
    COBYLA,
    COBYQA,
    Dogleg,
    LBFGSB,
    NewtonCG,
    NonlinearCG,
    SLSQP,
    SR1,
    TNC,
    TrustConstr,
    TrustKrylov,
    TrustNCG,
)

ALGORITHMS = [
    # Optax
    OptaxAdam,
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxAdafactor,
    OptaxAMSGrad,
    OptaxAdaGrad,
    OptaxAdaDelta,
    OptaxAdaMax,
    OptaxAdaMaxW,
    OptaxAdan,
    OptaxLion,
    OptaxLAMB,
    OptaxNadam,
    OptaxNadamW,
    OptaxRMSProp,
    OptaxRProp,
    OptaxRAdam,
    OptaxSGD,
    OptaxSGDM,
    OptaxNAG,
    OptaxNoisySGD,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxLookahead,
    OptaxScheduleFreeAdam,
    OptaxYogi,
    OptaxNovoGrad,
    OptaxOGD,
    OptaxOAdam,
    OptaxSignSGD,
    OptaxSignum,
    OptaxSM3,
    OptaxLBFGS,
    # SciPy
    BFGS,
    COBYLA,
    COBYQA,
    Dogleg,
    LBFGSB,
    NewtonCG,
    NonlinearCG,
    SLSQP,
    SR1,
    TNC,
    TrustConstr,
    TrustKrylov,
    TrustNCG,
]

# ── CLI ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="UIFO sweep over Optax + SciPy algorithms."
)
parser.add_argument(
    "-a",
    "--algo",
    required=True,
    help=f"Algorithm index (0–{len(ALGORITHMS) - 1}), or 'list' to print the table.",
)
parser.add_argument("-s", "--seed", type=int, default=0, help="Run seed (0–24).")
args = parser.parse_args()

if args.algo == "list":
    print(f"{'Index':<6} {'Class':<30} {'algorithm_str'}")
    print("-" * 60)
    for i, cls in enumerate(ALGORITHMS):
        print(f"{i:<6} {cls.__name__:<30} {cls.algorithm_str}")
    sys.exit(0)

algo_idx = int(args.algo)
if algo_idx < 0 or algo_idx >= len(ALGORITHMS):
    print(f"Error: --algo must be 0–{len(ALGORITHMS) - 1}, got {algo_idx}")
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
