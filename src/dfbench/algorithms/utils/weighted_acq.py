from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from botorch.acquisition import AcquisitionFunction


class WeightedAcquisitionFunction(AcquisitionFunction):
    """Weighted acquisition function from the πBO paper.
    https://arxiv.org/abs/2204.11051
    """

    pass
