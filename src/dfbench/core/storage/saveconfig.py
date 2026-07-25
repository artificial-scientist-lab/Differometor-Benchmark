"""Declarative save configuration for :class:`Objective`.

A :class:`SaveConfig` records *what* histories an Objective records during
a run. It replaces the nine individual boolean constructor arguments with
three standard flags plus a declarative ``save`` list of string tokens for
advanced combinations. The config is embedded in :class:`RunMetadata` so a
checkpoint records which histories were active, preventing silent
mismatches on resume.

Valid ``save`` tokens
--------------------

Standard histories:

| Token              | Effect                                                              |
|--------------------|---------------------------------------------------------------------|
| ``"grad"``         | Record gradients (one entry per logged call; batches reduced)       |
| ``"hessian"``      | Record Hessians (one entry per logged call; batches reduced)        |
| ``"eval_type"``    | Record one call-type bitmask per logged call                        |
| ``"batched_loss"`` | Store full ``(batch,)`` loss vectors instead of batch min           |
| ``"batched_grad"`` | Store full ``(batch, n_params)`` gradient arrays                    |
| ``"batched_hessian"``| Store full ``(batch, n_params, n_params)`` Hessian arrays       |
| ``"batched"``      | Convenience alias: expands to ``batched_loss``, ``batched_grad``,  |
|                    | ``batched_hessian``                                                |

Aux diagnostics (on problems that expose ``objective_function_aux``).
Selecting any aux token makes the Objective use that aux objective family
for value-bearing evaluation methods while preserving standard return
signatures. Gradient-only and Hessian-only methods differentiate the scalar
objective without producing a primal aux pytree, so selected aux histories
receive ``None`` placeholders:

| Token                       | Effect                                                                |
|-----------------------------|-----------------------------------------------------------------------|
| ``"sensitivity_loss"``      | Record unpenalised loss per logged call (reduced; else ``None``).      |
| ``"penalty"``               | Record summed penalty per logged call (reduced; else ``None``).       |
| ``"is_feasible"``           | Record feasibility per logged call (reduced; else ``None``).          |
| ``"power_values"``          | Record group powers per logged call (reduced; else ``None``).         |
| ``"violations"``            | Record violations per logged call (reduced; else ``None``).           |
| ``"aux"``                   | Convenience alias: expands to the five non-batched aux tokens above.  |
| ``"batched_sensitivity_loss"`` | Store full batched sensitivity loss arrays.                       |
| ``"batched_penalty"``       | Store full batched penalty arrays.                                    |
| ``"batched_is_feasible"``   | Store full batched feasibility bool arrays.                           |
| ``"batched_power_values"``  | Store full batched per-group power arrays.                            |
| ``"batched_violations"``    | Store full batched per-constraint violation arrays.                   |
| ``"batched_aux"``           | Convenience alias: expands to the five batched aux tokens above.      |

When a ``batched_*`` aux token is off and the corresponding non-batched
token is on, batched aux entries are reduced to the same representative
point used for parameters, gradients, and Hessians.

The three standard flags (``save_time_steps``, ``save_params_history``,
``save_batched_params_history``) remain explicit constructor arguments.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields

_AUX_FIELDS: tuple[str, ...] = (
    "sensitivity_loss",
    "penalty",
    "is_feasible",
    "power_values",
    "violations",
)

_ALIAS_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "batched": ("batched_loss", "batched_grad", "batched_hessian"),
    "aux": _AUX_FIELDS,
    "batched_aux": tuple(f"batched_{name}" for name in _AUX_FIELDS),
}

_DIRECT_TOKENS: frozenset[str] = frozenset(
    {
        "grad",
        "hessian",
        "eval_type",
        "batched_loss",
        "batched_grad",
        "batched_hessian",
        *_AUX_FIELDS,
        *(f"batched_{name}" for name in _AUX_FIELDS),
    }
)

VALID_TOKENS: frozenset[str] = _DIRECT_TOKENS | frozenset(_ALIAS_EXPANSIONS)

_LEGACY_FIELDS: dict[str, str] = {
    "batched_loss": "batched_losses",
    "batched_grad": "batched_grads",
    "batched_hessian": "batched_hessians",
    "batched_param": "batched_params",
}


@dataclass
class SaveConfig:
    """Declarative record of which histories an Objective tracks."""

    time_steps: bool = True
    params: bool = True
    grad: bool = False
    hessian: bool = False
    eval_type: bool = False
    batched_loss: bool = False
    batched_grad: bool = False
    batched_hessian: bool = False
    batched_param: bool = False
    sensitivity_loss: bool = False
    penalty: bool = False
    is_feasible: bool = False
    power_values: bool = False
    violations: bool = False
    batched_sensitivity_loss: bool = False
    batched_penalty: bool = False
    batched_is_feasible: bool = False
    batched_power_values: bool = False
    batched_violations: bool = False

    @property
    def has_aux(self) -> bool:
        """Whether any reduced or full-batch aux history is selected."""
        return any(self.wants_aux(name) for name in _AUX_FIELDS)

    def wants_aux(self, name: str) -> bool:
        """Whether an aux field is selected in either storage mode."""
        self._validate_aux_name(name)
        return bool(getattr(self, name) or getattr(self, f"batched_{name}"))

    def wants_batched_aux(self, name: str) -> bool:
        """Whether an aux field should retain its leading batch axis."""
        self._validate_aux_name(name)
        return bool(getattr(self, f"batched_{name}"))

    @staticmethod
    def _validate_aux_name(name: str) -> None:
        if name not in _AUX_FIELDS:
            raise ValueError(f"Unknown aux field {name!r}.")

    @classmethod
    def from_flags(
        cls,
        save_time_steps: bool = True,
        save_params_history: bool = True,
        save_batched_params_history: bool = False,
        save: list[str] | None = None,
    ) -> "SaveConfig":
        """Build a config from the common flags and declarative tokens."""
        config = cls(
            time_steps=save_time_steps,
            params=save_params_history,
            batched_param=save_batched_params_history,
        )
        for token in save or ():
            if token not in VALID_TOKENS:
                raise ValueError(
                    f"Unknown save token '{token}'. "
                    f"Valid tokens: {sorted(VALID_TOKENS)}."
                )
            for field_name in _ALIAS_EXPANSIONS.get(token, (token,)):
                setattr(config, field_name, True)
        return config

    def to_dict(self) -> dict[str, bool]:
        """Serialize to a plain dict for embedding in :class:`RunMetadata`."""
        return {field.name: bool(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "SaveConfig":
        """Reconstruct from a dict, accepting legacy plural field names."""
        values: dict[str, bool] = {}
        for field in fields(cls):
            default = field.default if field.default is not MISSING else False
            legacy_name = _LEGACY_FIELDS.get(field.name)
            value = data.get(
                field.name,
                data.get(legacy_name, default) if legacy_name else default,
            )
            values[field.name] = bool(value)
        return cls(**values)

    def mismatch(self, other: "SaveConfig") -> list[str]:
        """Return field names whose values differ from another config."""
        return [
            field.name
            for field in fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        ]
