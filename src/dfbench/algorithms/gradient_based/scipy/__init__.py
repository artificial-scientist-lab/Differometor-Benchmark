"""SciPy-based gradient optimizers."""

from dfbench.algorithms.gradient_based.scipy.bfgs import BFGS
from dfbench.algorithms.gradient_based.scipy.cobyla import COBYLA
from dfbench.algorithms.gradient_based.scipy.cobyqa import COBYQA
from dfbench.algorithms.gradient_based.scipy.dogleg import Dogleg
from dfbench.algorithms.gradient_based.scipy.lbfgsb import LBFGSB
from dfbench.algorithms.gradient_based.scipy.newton_cg import NewtonCG
from dfbench.algorithms.gradient_based.scipy.nonlinear_cg import NonlinearCG
from dfbench.algorithms.gradient_based.scipy.slsqp import SLSQP
from dfbench.algorithms.gradient_based.scipy.sr1 import SR1
from dfbench.algorithms.gradient_based.scipy.tnc import TNC
from dfbench.algorithms.gradient_based.scipy.trust_constr import TrustConstr
from dfbench.algorithms.gradient_based.scipy.trust_krylov import TrustKrylov
from dfbench.algorithms.gradient_based.scipy.trust_ncg import TrustNCG

__all__ = [
    "BFGS",
    "COBYLA",
    "COBYQA",
    "Dogleg",
    "LBFGSB",
    "NewtonCG",
    "NonlinearCG",
    "SLSQP",
    "SR1",
    "TNC",
    "TrustConstr",
    "TrustKrylov",
    "TrustNCG",
]
