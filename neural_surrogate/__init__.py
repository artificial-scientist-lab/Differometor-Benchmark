"""Neural surrogate models and training utilities."""

from .encodings import (
    CampaignEncoder,
    CampaignSample,
    EncodedCampaignDataset,
    make_campaign_dataset,
)
from .eval import evaluate, predict
from .model import (
    NeuralSurrogate,
    TransformerEncoderConfig,
    TransformerEncoderSurrogate,
)
from .train import TrainConfig, TrainHistory, fit, train_epoch

__all__ = [
    "CampaignEncoder",
    "CampaignSample",
    "EncodedCampaignDataset",
    "NeuralSurrogate",
    "TrainConfig",
    "TrainHistory",
    "TransformerEncoderConfig",
    "TransformerEncoderSurrogate",
    "evaluate",
    "fit",
    "make_campaign_dataset",
    "predict",
    "train_epoch",
]
