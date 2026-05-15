"""Encoding utilities for optimized campaign surrogate data.

Campaign samples are stored as H5 run payloads with:

- discrete topology in ``setup_graph``
- continuous optimization values in ``best_parameters``
- scalar supervised targets in ``loss_*`` keys

This module turns those payloads into fixed-width tensors usable by the simple
transformer surrogate in :mod:`neural_surrogate.model`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


TopologyStrategy = Literal["hashing", "vocabulary", "exact"]
ParameterStrategy = Literal["identity", "standard", "bounds"]


@dataclass(frozen=True)
class CampaignSample:
    """One decoded run/simplification sample from a campaign file."""

    setup_graph: Any
    best_parameters: Any
    losses: Mapping[str, float]
    experiment_data: Mapping[str, Any]
    source: str


@dataclass(frozen=True)
class EncodedCampaignSample:
    """Tensor-ready sample consumed by the training/eval loops."""

    x: torch.Tensor
    y: torch.Tensor
    source: str


class HashingTopologyEncoder:
    """Encode topology tokens into a fixed-width signed hashing vector."""

    def __init__(self, dim: int = 512, *, normalize: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = dim
        self.normalize = normalize

    @property
    def output_dim(self) -> int:
        return self.dim

    def fit(self, setup_graphs: Iterable[Any]) -> "HashingTopologyEncoder":
        return self

    def encode(self, setup_graph: Any) -> torch.Tensor:
        vector = torch.zeros(self.dim, dtype=torch.float32)
        for token in topology_tokens(setup_graph):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector) if self.normalize else vector


class VocabularyTopologyEncoder:
    """Encode topology token counts with a fitted vocabulary."""

    def __init__(
        self,
        max_tokens: int = 1024,
        *,
        min_count: int = 1,
        normalize: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if min_count <= 0:
            raise ValueError("min_count must be positive.")
        self.max_tokens = max_tokens
        self.min_count = min_count
        self.normalize = normalize
        self.token_to_index: dict[str, int] = {}

    @property
    def output_dim(self) -> int:
        return self.max_tokens + 1

    def fit(self, setup_graphs: Iterable[Any]) -> "VocabularyTopologyEncoder":
        counts: Counter[str] = Counter()
        for setup_graph in setup_graphs:
            counts.update(topology_tokens(setup_graph))

        selected = [
            token
            for token, count in counts.most_common(self.max_tokens)
            if count >= self.min_count
        ]
        self.token_to_index = {token: idx for idx, token in enumerate(selected)}
        return self

    def encode(self, setup_graph: Any) -> torch.Tensor:
        vector = torch.zeros(self.output_dim, dtype=torch.float32)
        unknown_index = self.max_tokens
        for token in topology_tokens(setup_graph):
            vector[self.token_to_index.get(token, unknown_index)] += 1.0
        return _l2_normalize(vector) if self.normalize else vector


class ExactTopologyEncoder:
    """One-hot encode complete topology identities observed during fitting."""

    def __init__(
        self,
        *,
        max_topologies: int | None = None,
        unknown_bucket: bool = True,
    ) -> None:
        if max_topologies is not None and max_topologies <= 0:
            raise ValueError("max_topologies must be positive when provided.")
        self.max_topologies = max_topologies
        self.unknown_bucket = unknown_bucket
        self.topology_to_index: dict[str, int] = {}

    @property
    def output_dim(self) -> int:
        return len(self.topology_to_index) + int(self.unknown_bucket)

    def fit(self, setup_graphs: Iterable[Any]) -> "ExactTopologyEncoder":
        self.topology_to_index = {}
        for setup_graph in setup_graphs:
            key = canonical_json(setup_graph)
            if key not in self.topology_to_index:
                if (
                    self.max_topologies is not None
                    and len(self.topology_to_index) >= self.max_topologies
                ):
                    continue
                self.topology_to_index[key] = len(self.topology_to_index)
        return self

    def encode(self, setup_graph: Any) -> torch.Tensor:
        vector = torch.zeros(self.output_dim, dtype=torch.float32)
        key = canonical_json(setup_graph)
        index = self.topology_to_index.get(key)
        if index is not None:
            vector[index] = 1.0
        elif self.unknown_bucket:
            vector[-1] = 1.0
        return vector


class ParameterEncoder:
    """Flatten and scale continuous ``best_parameters`` values."""

    def __init__(
        self,
        strategy: ParameterStrategy = "standard",
        *,
        size: int | None = None,
        fill_value: float = 0.0,
    ) -> None:
        self.strategy = strategy
        self.size = size
        self.fill_value = fill_value
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.low: torch.Tensor | None = None
        self.high: torch.Tensor | None = None

    @property
    def output_dim(self) -> int:
        if self.size is None:
            raise RuntimeError("ParameterEncoder must be fit before output_dim is known.")
        return self.size

    def fit(self, samples: Sequence[CampaignSample]) -> "ParameterEncoder":
        flattened = [flatten_parameters(sample.best_parameters) for sample in samples]
        self.size = self.size or max((len(values) for values in flattened), default=0)
        if self.size <= 0:
            raise ValueError("No best_parameters values found to fit.")

        matrix = torch.stack(
            [pad_or_truncate(values, self.size, self.fill_value) for values in flattened]
        )
        if self.strategy == "standard":
            self.mean = matrix.mean(dim=0)
            self.std = matrix.std(dim=0, unbiased=False).clamp_min(1e-8)
        elif self.strategy == "bounds":
            self.low, self.high = self._fit_bounds(samples, matrix)
        elif self.strategy != "identity":
            raise ValueError(f"Unknown parameter encoding strategy: {self.strategy}")
        return self

    def encode(self, best_parameters: Any) -> torch.Tensor:
        values = pad_or_truncate(
            flatten_parameters(best_parameters),
            self.output_dim,
            self.fill_value,
        )
        if self.strategy == "identity":
            return values
        if self.strategy == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("ParameterEncoder has not been fit.")
            return (values - self.mean) / self.std
        if self.strategy == "bounds":
            if self.low is None or self.high is None:
                raise RuntimeError("ParameterEncoder has not been fit.")
            return (values - self.low) / (self.high - self.low).clamp_min(1e-8)
        raise ValueError(f"Unknown parameter encoding strategy: {self.strategy}")

    def _fit_bounds(
        self,
        samples: Sequence[CampaignSample],
        matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bounds = first_parameter_bounds(samples)
        if not bounds:
            return matrix.min(dim=0).values, matrix.max(dim=0).values

        lows: list[float] = []
        highs: list[float] = []
        for low, high in bounds:
            lows.append(float(low))
            highs.append(float(high))

        lows = repeat_to_size(lows, self.output_dim)
        highs = repeat_to_size(highs, self.output_dim)
        low_tensor = pad_or_truncate(lows, self.output_dim, float(matrix.min().item()))
        high_tensor = pad_or_truncate(highs, self.output_dim, float(matrix.max().item()))
        invalid = high_tensor <= low_tensor
        if bool(invalid.any()):
            observed_low = matrix.min(dim=0).values
            observed_high = matrix.max(dim=0).values
            low_tensor = torch.where(invalid, observed_low, low_tensor)
            high_tensor = torch.where(invalid, observed_high, high_tensor)
        return low_tensor, high_tensor


class CampaignEncoder:
    """Joint encoder for topology, parameters, and one scalar loss target."""

    def __init__(
        self,
        *,
        topology_strategy: TopologyStrategy = "hashing",
        parameter_strategy: ParameterStrategy = "standard",
        topology_dim: int = 512,
        loss_key: str = "loss_senspow",
        fallback_loss_keys: Sequence[str] = ("loss_incoherent_senspow", "loss_optimized"),
    ) -> None:
        self.loss_key = loss_key
        self.fallback_loss_keys = tuple(fallback_loss_keys)
        self.topology_encoder = make_topology_encoder(
            topology_strategy,
            dim=topology_dim,
        )
        self.parameter_encoder = ParameterEncoder(parameter_strategy)

    @property
    def input_dim(self) -> int:
        return self.topology_encoder.output_dim + self.parameter_encoder.output_dim

    def fit(self, samples: Sequence[CampaignSample]) -> "CampaignEncoder":
        if not samples:
            raise ValueError("Cannot fit CampaignEncoder with no samples.")
        self.topology_encoder.fit(sample.setup_graph for sample in samples)
        self.parameter_encoder.fit(samples)
        return self

    def encode(self, sample: CampaignSample) -> EncodedCampaignSample:
        topology = self.topology_encoder.encode(sample.setup_graph)
        parameters = self.parameter_encoder.encode(sample.best_parameters)
        target = torch.tensor(
            [select_loss(sample.losses, self.loss_key, self.fallback_loss_keys)]
        )
        return EncodedCampaignSample(
            x=torch.cat([topology, parameters]).to(torch.float32),
            y=target.to(torch.float32),
            source=sample.source,
        )


class EncodedCampaignDataset(Dataset):
    """Torch dataset of encoded campaign samples."""

    def __init__(
        self,
        samples: Sequence[CampaignSample],
        encoder: CampaignEncoder,
        *,
        fit_encoder: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.encoder = encoder
        if fit_encoder:
            self.encoder.fit(self.samples)
        self.encoded = [self.encoder.encode(sample) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.encoded[index]
        return {"x": sample.x, "y": sample.y}

    @property
    def sources(self) -> list[str]:
        return [sample.source for sample in self.encoded]


def make_topology_encoder(
    strategy: TopologyStrategy,
    *,
    dim: int = 512,
) -> HashingTopologyEncoder | VocabularyTopologyEncoder | ExactTopologyEncoder:
    if strategy == "hashing":
        return HashingTopologyEncoder(dim=dim)
    if strategy == "vocabulary":
        return VocabularyTopologyEncoder(max_tokens=dim)
    if strategy == "exact":
        return ExactTopologyEncoder(max_topologies=dim)
    raise ValueError(f"Unknown topology encoding strategy: {strategy}")


def load_campaign_samples(
    paths: str | Path | Iterable[str | Path],
    *,
    loss_key: str = "loss_senspow",
    fallback_loss_keys: Sequence[str] = (
        "loss_incoherent_senspow",
        "loss_optimized",
    ),
    include_simplifications: bool = False,
    num_workers: int = 0,
) -> list[CampaignSample]:
    """Load H5 campaign runs that have topology, parameters, and target loss."""
    h5_paths = _normalize_paths(paths)
    if num_workers < 0:
        num_workers = min(len(h5_paths), os.cpu_count() or 1)
    if num_workers <= 1 or len(h5_paths) <= 1:
        return _load_campaign_samples_serial(
            h5_paths,
            loss_key=loss_key,
            fallback_loss_keys=fallback_loss_keys,
            include_simplifications=include_simplifications,
        )

    samples: list[CampaignSample] = []
    worker_args = [
        (h5_path, loss_key, tuple(fallback_loss_keys), include_simplifications)
        for h5_path in h5_paths
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for h5_path, file_samples, error in executor.map(
            _load_campaign_samples_worker,
            worker_args,
        ):
            if error is not None:
                warnings.warn(f"Skipping unreadable H5 file {h5_path}: {error}")
                continue
            samples.extend(file_samples)
    return samples


def _load_campaign_samples_serial(
    h5_paths: Sequence[Path],
    *,
    loss_key: str,
    fallback_loss_keys: Sequence[str],
    include_simplifications: bool,
) -> list[CampaignSample]:
    samples: list[CampaignSample] = []
    for h5_path in h5_paths:
        try:
            samples.extend(
                read_campaign_samples_h5(
                    h5_path,
                    loss_key=loss_key,
                    fallback_loss_keys=fallback_loss_keys,
                    include_simplifications=include_simplifications,
                )
            )
        except OSError as exc:
            warnings.warn(f"Skipping unreadable H5 file {h5_path}: {exc}")
    return samples


def _load_campaign_samples_worker(
    args: tuple[Path, str, tuple[str, ...], bool],
) -> tuple[Path, list[CampaignSample], str | None]:
    h5_path, loss_key, fallback_loss_keys, include_simplifications = args
    try:
        samples = read_campaign_samples_h5(
            h5_path,
            loss_key=loss_key,
            fallback_loss_keys=fallback_loss_keys,
            include_simplifications=include_simplifications,
        )
    except OSError as exc:
        return h5_path, [], str(exc)
    return h5_path, samples, None


def make_campaign_dataset(
    paths: str | Path | Iterable[str | Path],
    *,
    topology_strategy: TopologyStrategy = "hashing",
    parameter_strategy: ParameterStrategy = "standard",
    topology_dim: int = 512,
    loss_key: str = "loss_senspow",
    fallback_loss_keys: Sequence[str] = (
        "loss_incoherent_senspow",
        "loss_optimized",
    ),
    include_simplifications: bool = False,
    num_workers: int = 0,
) -> EncodedCampaignDataset:
    """Load and encode campaign data for surrogate training."""
    samples = load_campaign_samples(
        paths,
        loss_key=loss_key,
        fallback_loss_keys=fallback_loss_keys,
        include_simplifications=include_simplifications,
        num_workers=num_workers,
    )
    encoder = CampaignEncoder(
        topology_strategy=topology_strategy,
        parameter_strategy=parameter_strategy,
        topology_dim=topology_dim,
        loss_key=loss_key,
        fallback_loss_keys=fallback_loss_keys,
    )
    return EncodedCampaignDataset(samples, encoder)


def samples_from_payload(
    payload: Mapping[str, Any],
    *,
    experiment_data: Mapping[str, Any],
    source: str,
    loss_key: str,
    fallback_loss_keys: Sequence[str],
) -> list[CampaignSample]:
    if "setup_graph" not in payload or "best_parameters" not in payload:
        return []
    losses = scalar_losses(payload)
    try:
        select_loss(losses, loss_key, fallback_loss_keys)
    except KeyError:
        return []
    return [
        CampaignSample(
            setup_graph=payload["setup_graph"],
            best_parameters=payload["best_parameters"],
            losses=losses,
            experiment_data=experiment_data,
            source=source,
        )
    ]


def simplification_samples(
    payload: Mapping[str, Any],
    *,
    experiment_data: Mapping[str, Any],
    source_prefix: str,
    loss_key: str,
    fallback_loss_keys: Sequence[str],
) -> list[CampaignSample]:
    simplifications = payload.get("simplifications")
    if not isinstance(simplifications, Mapping):
        return []

    samples: list[CampaignSample] = []
    for strategy, threshold_bucket in simplifications.items():
        if not isinstance(threshold_bucket, Mapping):
            continue
        for threshold, simplified_payload in threshold_bucket.items():
            if not isinstance(simplified_payload, Mapping):
                continue
            samples.extend(
                samples_from_payload(
                    simplified_payload,
                    experiment_data=experiment_data,
                    source=f"{source_prefix}:simplifications/{strategy}/{threshold}",
                    loss_key=loss_key,
                    fallback_loss_keys=fallback_loss_keys,
                )
            )
    return samples


def topology_tokens(value: Any, *, prefix: str = "setup") -> list[str]:
    """Extract stable categorical tokens from a setup graph."""
    if isinstance(value, str):
        value = _maybe_json(value)

    tokens: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value.keys(), key=str):
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}"
            tokens.append(f"key:{child_prefix}")
            tokens.extend(topology_tokens(value[key], prefix=child_prefix))
        return tokens

    if isinstance(value, (list, tuple)):
        tokens.append(f"list:{prefix}:len={len(value)}")
        for index, item in enumerate(value):
            tokens.extend(topology_tokens(item, prefix=f"{prefix}[]"))
            if isinstance(item, Mapping):
                type_value = item.get("type") or item.get("name") or item.get("kind")
                if type_value is not None:
                    tokens.append(f"seq:{prefix}:{index}:{type_value}")
        return tokens

    if isinstance(value, (bool, np.bool_)):
        return [f"value:{prefix}:{bool(value)}"]
    if isinstance(value, (int, np.integer)):
        return [f"value:{prefix}:{int(value)}"]
    if isinstance(value, (float, np.floating)):
        return [f"number:{prefix}"]
    if value is None:
        return [f"none:{prefix}"]
    return [f"value:{prefix}:{str(value)}"]


def flatten_parameters(best_parameters: Any) -> list[float]:
    """Flatten scalar/list/matrix parameters into a continuous vector."""
    if best_parameters is None:
        return []
    if isinstance(best_parameters, str):
        best_parameters = _maybe_json(best_parameters)
    array = np.asarray(best_parameters, dtype=np.float32)
    if array.shape == ():
        return [float(array.item())]
    return [float(value) for value in array.reshape(-1).tolist()]


def scalar_losses(payload: Mapping[str, Any]) -> dict[str, float]:
    losses: dict[str, float] = {}
    for key, value in payload.items():
        key_text = str(key)
        if not key_text.startswith("loss_"):
            continue
        scalar = as_float_scalar(value)
        if scalar is not None and math.isfinite(scalar):
            losses[key_text] = scalar
    return losses


def select_loss(
    losses: Mapping[str, float],
    loss_key: str,
    fallback_loss_keys: Sequence[str] = (),
) -> float:
    for key in (loss_key, *fallback_loss_keys):
        if key in losses:
            return float(losses[key])
    available = sorted(losses.keys())
    raise KeyError(
        f"No requested loss key found. Requested {loss_key}; available: {available}"
    )


def first_parameter_bounds(samples: Sequence[CampaignSample]) -> list[tuple[float, float]]:
    for sample in samples:
        bounds = parameter_bounds_from_experiment(sample.experiment_data)
        if bounds:
            return bounds
    return []


def parameter_bounds_from_experiment(
    experiment_data: Mapping[str, Any],
) -> list[tuple[float, float]]:
    """Return parameter bounds using ``optimized_properties`` order when present."""
    raw_bounds = experiment_data.get("parameter_type_bounds")
    if not isinstance(raw_bounds, Mapping):
        return flatten_bounds(raw_bounds)

    optimized_properties = experiment_data.get("optimized_properties")
    property_names = _as_string_list(optimized_properties)
    if not property_names:
        return flatten_bounds(raw_bounds)

    ordered_bounds: list[tuple[float, float]] = []
    for property_name in property_names:
        if property_name in raw_bounds:
            ordered_bounds.extend(flatten_bounds(raw_bounds[property_name]))
    return ordered_bounds or flatten_bounds(raw_bounds)


def flatten_bounds(raw_bounds: Any) -> list[tuple[float, float]]:
    """Flatten experiment-level ``parameter_type_bounds`` into low/high pairs."""
    if raw_bounds is None:
        return []
    if isinstance(raw_bounds, str):
        raw_bounds = _maybe_json(raw_bounds)

    pairs: list[tuple[float, float]] = []
    if isinstance(raw_bounds, Mapping):
        for key in sorted(raw_bounds.keys(), key=str):
            pairs.extend(flatten_bounds(raw_bounds[key]))
        return pairs

    if isinstance(raw_bounds, (list, tuple, np.ndarray)):
        values = list(raw_bounds)
        if len(values) == 2 and all(as_float_scalar(value) is not None for value in values):
            low = as_float_scalar(values[0])
            high = as_float_scalar(values[1])
            if low is not None and high is not None:
                return [(low, high)]
        for value in values:
            pairs.extend(flatten_bounds(value))
    return pairs


def pad_or_truncate(
    values: Sequence[float],
    size: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    out = torch.full((size,), float(fill_value), dtype=torch.float32)
    if size <= 0:
        return out
    count = min(len(values), size)
    if count:
        out[:count] = torch.tensor(values[:count], dtype=torch.float32)
    return out


def repeat_to_size(values: Sequence[float], size: int) -> list[float]:
    if not values or size <= 0:
        return []
    repeats = math.ceil(size / len(values))
    return list(values) * repeats


def canonical_json(value: Any) -> str:
    if isinstance(value, str):
        value = _maybe_json(value)
    return json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"))


def read_campaign_h5(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Decode ``/experiment_data`` and ``/runs`` from a campaign H5 file."""
    import h5py

    h5_path = Path(path)
    with h5py.File(h5_path, "r") as handle:
        if "experiment_data" not in handle or "runs" not in handle:
            raise RuntimeError(f"{h5_path} must contain /experiment_data and /runs.")

        experiment_data = read_h5_json(handle["experiment_data"])
        runs_raw = read_h5_json(handle["runs"])
        if not isinstance(experiment_data, dict):
            raise RuntimeError(f"{h5_path} /experiment_data did not decode to a dict.")
        if not isinstance(runs_raw, dict):
            raise RuntimeError(f"{h5_path} /runs did not decode to a dict.")

    runs: dict[str, dict[str, Any]] = {}
    for run_id, payload in runs_raw.items():
        if isinstance(payload, dict):
            runs[str(run_id)] = payload
    return experiment_data, runs


def read_campaign_samples_h5(
    path: str | Path,
    *,
    loss_key: str = "loss_senspow",
    fallback_loss_keys: Sequence[str] = ("loss_incoherent_senspow", "loss_optimized"),
    include_simplifications: bool = False,
) -> list[CampaignSample]:
    """Decode only trainable fields from a campaign H5 file."""
    import h5py

    h5_path = Path(path)
    samples: list[CampaignSample] = []
    with h5py.File(h5_path, "r") as handle:
        if "experiment_data" not in handle or "runs" not in handle:
            raise RuntimeError(f"{h5_path} must contain /experiment_data and /runs.")

        experiment_data = read_h5_json(handle["experiment_data"])
        if not isinstance(experiment_data, dict):
            raise RuntimeError(f"{h5_path} /experiment_data did not decode to a dict.")

        runs_group = handle["runs"]
        for run_id in sorted(runs_group.keys(), key=_h5_sort_key):
            run_group = runs_group[run_id]
            payload = read_training_payload(run_group)
            samples.extend(
                samples_from_payload(
                    payload,
                    experiment_data=experiment_data,
                    source=f"{h5_path}:{run_id}",
                    loss_key=loss_key,
                    fallback_loss_keys=fallback_loss_keys,
                )
            )

            if include_simplifications and "simplifications" in run_group:
                simplifications = run_group["simplifications"]
                samples.extend(
                    read_simplification_samples_h5(
                        simplifications,
                        experiment_data=experiment_data,
                        source_prefix=f"{h5_path}:{run_id}",
                        loss_key=loss_key,
                        fallback_loss_keys=fallback_loss_keys,
                    )
                )
    return samples


def read_training_payload(group: Any) -> dict[str, Any]:
    """Read only topology, parameters, and scalar losses from one H5 group."""
    payload: dict[str, Any] = {}
    for key in ("setup_graph", "best_parameters"):
        if key in group:
            payload[key] = read_h5_json(group[key])

    for key in group.keys():
        key_text = str(key)
        if key_text.startswith("loss_"):
            payload[key_text] = read_h5_json(group[key])
    return payload


def read_simplification_samples_h5(
    simplifications_group: Any,
    *,
    experiment_data: Mapping[str, Any],
    source_prefix: str,
    loss_key: str,
    fallback_loss_keys: Sequence[str],
) -> list[CampaignSample]:
    samples: list[CampaignSample] = []
    for strategy in sorted(simplifications_group.keys(), key=str):
        strategy_group = simplifications_group[strategy]
        for threshold in sorted(strategy_group.keys(), key=_h5_sort_key):
            payload = read_training_payload(strategy_group[threshold])
            samples.extend(
                samples_from_payload(
                    payload,
                    experiment_data=experiment_data,
                    source=f"{source_prefix}:simplifications/{strategy}/{threshold}",
                    loss_key=loss_key,
                    fallback_loss_keys=fallback_loss_keys,
                )
            )
    return samples


def read_h5_json(node: Any) -> Any:
    """Decode the JSON-like H5 representation used by campaign files."""
    import h5py

    if isinstance(node, h5py.Group):
        json_type = _attr_str(node, "__json_type__")
        if json_type == "list":
            keys = sorted(
                node.keys(),
                key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)),
            )
            return [read_h5_json(node[key]) for key in keys]

        out: dict[str, Any] = {}
        for key in node.keys():
            child = node[key]
            original_key = _attr_str(child, "__original_key__")
            out[original_key or key] = read_h5_json(child)
        return out

    if isinstance(node, h5py.Dataset):
        json_type = _attr_str(node, "__json_type__")
        raw = node[()]

        if json_type == "null":
            return None
        if json_type == "json_blob":
            return json.loads(_decode_scalar(raw))
        if json_type == "string":
            return str(_decode_scalar(raw))
        if json_type == "port_index_table":
            return _read_port_index_table(node)
        if json_type == "parameter_bounds_table":
            return _read_parameter_bounds_table(node)
        if json_type == "compressed_utf8_blob":
            return _read_compressed_utf8_blob(node)
        return _to_python(raw)

    raise TypeError(f"Unsupported HDF5 node type: {type(node)!r}")


def as_float_scalar(value: Any) -> float | None:
    if isinstance(value, str):
        parsed = _maybe_json(value)
        value = parsed
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
        else:
            return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _normalize_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidates = [Path(paths)]
    else:
        candidates = [Path(path) for path in paths]

    out: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            out.extend(sorted(candidate.glob("*.h5")))
        else:
            out.append(candidate)
    return out


def _h5_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = _maybe_json(value)
        if parsed is not value:
            return _as_string_list(parsed)
        return [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(_decode_scalar(item)) for item in value]
    return [str(_decode_scalar(value))]


def _maybe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_python(item) for item in value.tolist()]
    if isinstance(value, np.void) and value.dtype.names:
        return {name: _to_python(value[name]) for name in value.dtype.names}
    return _decode_scalar(value)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr_str(node: Any, name: str) -> str | None:
    if name not in node.attrs:
        return None
    value = _decode_scalar(node.attrs[name])
    return str(value) if value is not None else None


def _read_port_index_table(node: Any) -> dict[str, int]:
    table = node[()]
    if not isinstance(table, np.ndarray) or table.dtype.names is None:
        raise RuntimeError(f"port_index_table dataset is not structured: {node.name}")
    if "port" not in table.dtype.names or "index" not in table.dtype.names:
        raise RuntimeError(
            f"port_index_table dataset missing required fields: {node.name}"
        )

    out: dict[str, int] = {}
    for row in table:
        port_raw = row["port"]
        idx_raw = row["index"]
        if isinstance(port_raw, (bytes, np.bytes_)):
            port = bytes(port_raw).rstrip(b"\x00").decode("utf-8", errors="replace")
        else:
            port = str(_decode_scalar(port_raw))
        out[port] = int(_decode_scalar(idx_raw))
    return out


def _read_parameter_bounds_table(node: Any) -> dict[str, tuple[float, float]]:
    table = node[()]
    if not isinstance(table, np.ndarray) or table.dtype.names is None:
        raise RuntimeError(
            f"parameter_bounds_table dataset is not structured: {node.name}"
        )

    names = set(table.dtype.names)
    name_field = "parameter_type" if "parameter_type" in names else "type"
    low_field = "low" if "low" in names else "lower"
    high_field = "high" if "high" in names else "upper"
    if name_field not in names or low_field not in names or high_field not in names:
        raise RuntimeError(
            f"parameter_bounds_table dataset missing required fields: {node.name}"
        )

    out: dict[str, tuple[float, float]] = {}
    for row in table:
        raw_name = row[name_field]
        if isinstance(raw_name, (bytes, np.bytes_)):
            name = bytes(raw_name).rstrip(b"\x00").decode("utf-8", errors="replace")
        else:
            name = str(_decode_scalar(raw_name))
        out[name] = (
            float(_decode_scalar(row[low_field])),
            float(_decode_scalar(row[high_field])),
        )
    return out


def _read_compressed_utf8_blob(node: Any) -> str:
    raw = node[()]
    if isinstance(raw, np.ndarray):
        payload = raw.tobytes()
    elif isinstance(raw, (bytes, bytearray)):
        payload = bytes(raw)
    else:
        payload = bytes(np.asarray(raw, dtype=np.uint8).tobytes())
    return zlib.decompress(payload).decode("utf-8")


def _l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm(p=2)
    if float(norm) == 0.0:
        return vector
    return vector / norm


__all__ = [
    "CampaignEncoder",
    "CampaignSample",
    "EncodedCampaignDataset",
    "EncodedCampaignSample",
    "ExactTopologyEncoder",
    "HashingTopologyEncoder",
    "ParameterEncoder",
    "VocabularyTopologyEncoder",
    "as_float_scalar",
    "canonical_json",
    "flatten_bounds",
    "flatten_parameters",
    "load_campaign_samples",
    "make_campaign_dataset",
    "make_topology_encoder",
    "parameter_bounds_from_experiment",
    "read_campaign_h5",
    "read_campaign_samples_h5",
    "read_h5_json",
    "read_training_payload",
    "repeat_to_size",
    "scalar_losses",
    "select_loss",
    "topology_tokens",
]
