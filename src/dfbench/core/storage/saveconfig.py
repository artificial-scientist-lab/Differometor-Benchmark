"""Declarative save configuration for :class:`Objective`.

A :class:`SaveConfig` records *what* histories an Objective records during
a run. It replaces the nine individual boolean constructor arguments with
two standard flags plus a declarative ``save`` list of string tokens for
advanced combinations. The config is embedded in :class:`RunMetadata` so a
checkpoint records which histories were active, preventing silent
mismatches on resume.

Valid ``save`` tokens
--------------------

Standard histories:

| Token              | Effect                                                              |
|--------------------|---------------------------------------------------------------------|
| ``"grad"``         | Record gradient history (reduced to one entry per eval for batches)|
| ``"hessian"``      | Record Hessian history (reduced to one entry per eval for batches) |
| ``"eval_type"``    | Record per-eval type bitmask history                                |
| ``"batched_loss"`` | Store full ``(batch,)`` loss vectors instead of batch min           |
| ``"batched_grad"`` | Store full ``(batch, n_params)`` gradient arrays                    |
| ``"batched_hessian"``| Store full ``(batch, n_params, n_params)`` Hessian arrays       |
| ``"batched"``      | Convenience alias: expands to ``batched_loss``, ``batched_grad``,  |
|                    | ``batched_hessian``                                                |

Aux diagnostics (only recorded by the ``value_aux`` / ``value_and_grad_aux``
/ ``vmap_*_aux`` methods, on problems that opt into the power-penalty
contract):

| Token                       | Effect                                                                |
|-----------------------------|-----------------------------------------------------------------------|
| ``"sensitivity_loss"``      | Record the unpenalised sensitivity loss per aux eval (reduced).       |
| ``"penalty"``               | Record the summed penalty per aux eval (reduced).                     |
| ``"is_feasible"``           | Record the physical feasibility flag per aux eval (reduced).          |
| ``"power_values"``          | Record per-group powers (hard, soft, detector) per aux eval (reduced).|
| ``"violations"``            | Record per-constraint penalty values per aux eval (reduced).          |
| ``"aux"``                   | Convenience alias: expands to the five non-batched aux tokens above.  |
| ``"batched_sensitivity_loss"`` | Store full batched sensitivity loss arrays.                       |
| ``"batched_penalty"``       | Store full batched penalty arrays.                                    |
| ``"batched_is_feasible"``   | Store full batched feasibility bool arrays.                           |
| ``"batched_power_values"``  | Store full batched per-group power arrays.                            |
| ``"batched_violations"``    | Store full batched per-constraint violation arrays.                   |
| ``"batched_aux"``           | Convenience alias: expands to the five batched aux tokens above.      |

When a ``batched_*`` aux token is off and the corresponding non-batched
token is on, batched aux entries are reduced to the representative point
(the index of the best loss within the batch), so the recorded
``is_feasible`` and ``violations`` reflect that best point. This matches
the reduction rule used for gradients and Hessians.

The two standard flags (``save_time_steps``, ``save_params_history``,
``save_batched_params_history``) remain as explicit booleans because they
are the most commonly toggled and have less combinatorial interaction with
the batched variants.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid advanced save tokens.
VALID_TOKENS: frozenset[str] = frozenset(
    {
        "grad",
        "hessian",
        "eval_type",
        "batched_loss",
        "batched_grad",
        "batched_hessian",
        "batched",  # convenience alias, expanded at construction
        # Aux diagnostics (non-batched / reduced):
        "sensitivity_loss",
        "penalty",
        "is_feasible",
        "power_values",
        "violations",
        "aux",  # convenience alias, expanded at construction
        # Aux diagnostics (batched / full):
        "batched_sensitivity_loss",
        "batched_penalty",
        "batched_is_feasible",
        "batched_power_values",
        "batched_violations",
        "batched_aux",  # convenience alias, expanded at construction
    }
)

# Expansion of the "batched" convenience alias.
_BATCHED_EXPANSION: list[str] = [
    "batched_loss",
    "batched_grad",
    "batched_hessian",
]

# Expansion of the "aux" convenience alias (non-batched aux diagnostics).
_AUX_EXPANSION: list[str] = [
    "sensitivity_loss",
    "penalty",
    "is_feasible",
    "power_values",
    "violations",
]

# Expansion of the "batched_aux" convenience alias.
_BATCHED_AUX_EXPANSION: list[str] = [
    "batched_sensitivity_loss",
    "batched_penalty",
    "batched_is_feasible",
    "batched_power_values",
    "batched_violations",
]

# Maps each non-batched aux token to its SaveConfig field name.
_AUX_FIELD_MAP: dict[str, str] = {
    "sensitivity_loss": "sensitivity_loss",
    "penalty": "penalty",
    "is_feasible": "is_feasible",
    "power_values": "power_values",
    "violations": "violations",
}

# Maps each batched aux token to its SaveConfig field name.
_BATCHED_AUX_FIELD_MAP: dict[str, str] = {
    "batched_sensitivity_loss": "batched_sensitivity_loss",
    "batched_penalty": "batched_penalty",
    "batched_is_feasible": "batched_is_feasible",
    "batched_power_values": "batched_power_values",
    "batched_violations": "batched_violations",
}


@dataclass
class SaveConfig:
    """Declarative record of which histories an Objective tracks.

    Attributes:
        time_steps: Record elapsed-time timestamps per evaluation.
        params: Record parameter vectors (reduced for batches).
        grad: Record gradient vectors (reduced for batches).
        hessian: Record Hessian matrices (reduced for batches).
        eval_type: Record per-eval type bitmask history.
        batched_loss: Store full batched loss vectors.
        batched_grad: Store full batched gradient arrays.
        batched_hessian: Store full batched Hessian arrays.
        batched_param: Store full batched parameter arrays.
        sensitivity_loss: Record the unpenalised sensitivity loss per aux eval.
        penalty: Record the summed penalty per aux eval.
        is_feasible: Record the physical feasibility flag per aux eval.
        power_values: Record per-group powers per aux eval.
        violations: Record per-constraint penalty values per aux eval.
        batched_sensitivity_loss: Store full batched sensitivity loss arrays.
        batched_penalty: Store full batched penalty arrays.
        batched_is_feasible: Store full batched feasibility bool arrays.
        batched_power_values: Store full batched per-group power arrays.
        batched_violations: Store full batched per-constraint violation arrays.
    """

    time_steps: bool = True
    params: bool = True
    grad: bool = False
    hessian: bool = False
    eval_type: bool = False
    batched_loss: bool = False
    batched_grad: bool = False
    batched_hessian: bool = False
    batched_param: bool = False
    # Aux diagnostics (non-batched / reduced)
    sensitivity_loss: bool = False
    penalty: bool = False
    is_feasible: bool = False
    power_values: bool = False
    violations: bool = False
    # Aux diagnostics (batched / full)
    batched_sensitivity_loss: bool = False
    batched_penalty: bool = False
    batched_is_feasible: bool = False
    batched_power_values: bool = False
    batched_violations: bool = False

    @classmethod
    def from_flags(
        cls,
        save_time_steps: bool = True,
        save_params_history: bool = True,
        save_batched_params_history: bool = False,
        save: list[str] | None = None,
    ) -> "SaveConfig":
        """Build a :class:`SaveConfig` from the standard flags + token list.

        Args:
            save_time_steps: Record timestamps.
            save_params_history: Record parameter history.
            save_batched_params_history: Store full ``(batch, n_params)``
                parameter arrays instead of the reduced representative
                point for batched evals.
            save: List of advanced tokens (see module docstring).

        Raises:
            ValueError: If an unknown token is encountered.
        """
        cfg = cls(
            time_steps=save_time_steps,
            params=save_params_history,
            batched_param=save_batched_params_history,
        )
        if save:
            expanded: list[str] = []
            for token in save:
                if token not in VALID_TOKENS:
                    raise ValueError(
                        f"Unknown save token '{token}'. "
                        f"Valid tokens: {sorted(VALID_TOKENS)}."
                    )
                if token == "batched":
                    expanded.extend(_BATCHED_EXPANSION)
                elif token == "aux":
                    expanded.extend(_AUX_EXPANSION)
                elif token == "batched_aux":
                    expanded.extend(_BATCHED_AUX_EXPANSION)
                else:
                    expanded.append(token)

            seen = set()
            for t in expanded:
                if t in seen:
                    continue
                seen.add(t)
                if t == "grad":
                    cfg.grad = True
                elif t == "hessian":
                    cfg.hessian = True
                elif t == "eval_type":
                    cfg.eval_type = True
                elif t == "batched_loss":
                    cfg.batched_loss = True
                elif t == "batched_grad":
                    cfg.batched_grad = True
                elif t == "batched_hessian":
                    cfg.batched_hessian = True
                elif t in _AUX_FIELD_MAP:
                    setattr(cfg, _AUX_FIELD_MAP[t], True)
                elif t in _BATCHED_AUX_FIELD_MAP:
                    setattr(cfg, _BATCHED_AUX_FIELD_MAP[t], True)

        return cfg

    def to_dict(self) -> dict[str, bool]:
        """Serialize to a plain dict for embedding in :class:`RunMetadata`."""
        return {
            "time_steps": self.time_steps,
            "params": self.params,
            "grad": self.grad,
            "hessian": self.hessian,
            "eval_type": self.eval_type,
            "batched_loss": self.batched_loss,
            "batched_grad": self.batched_grad,
            "batched_hessian": self.batched_hessian,
            "batched_param": self.batched_param,
            "sensitivity_loss": self.sensitivity_loss,
            "penalty": self.penalty,
            "is_feasible": self.is_feasible,
            "power_values": self.power_values,
            "violations": self.violations,
            "batched_sensitivity_loss": self.batched_sensitivity_loss,
            "batched_penalty": self.batched_penalty,
            "batched_is_feasible": self.batched_is_feasible,
            "batched_power_values": self.batched_power_values,
            "batched_violations": self.batched_violations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SaveConfig":
        """Reconstruct from a :meth:`to_dict` dict (missing keys -> defaults).

        Also accepts legacy plurals (``batched_losses`` etc.) for backward
        compatibility with checkpoints written by earlier versions.
        """
        return cls(
            time_steps=bool(d.get("time_steps", True)),
            params=bool(d.get("params", True)),
            grad=bool(d.get("grad", False)),
            hessian=bool(d.get("hessian", False)),
            eval_type=bool(d.get("eval_type", False)),
            batched_loss=bool(d.get("batched_loss", d.get("batched_losses", False))),
            batched_grad=bool(d.get("batched_grad", d.get("batched_grads", False))),
            batched_hessian=bool(
                d.get("batched_hessian", d.get("batched_hessians", False))
            ),
            batched_param=bool(d.get("batched_param", d.get("batched_params", False))),
            sensitivity_loss=bool(d.get("sensitivity_loss", False)),
            penalty=bool(d.get("penalty", False)),
            is_feasible=bool(d.get("is_feasible", False)),
            power_values=bool(d.get("power_values", False)),
            violations=bool(d.get("violations", False)),
            batched_sensitivity_loss=bool(d.get("batched_sensitivity_loss", False)),
            batched_penalty=bool(d.get("batched_penalty", False)),
            batched_is_feasible=bool(d.get("batched_is_feasible", False)),
            batched_power_values=bool(d.get("batched_power_values", False)),
            batched_violations=bool(d.get("batched_violations", False)),
        )

    def mismatch(self, other: "SaveConfig") -> list[str]:
        """Return a list of field names where ``self`` and ``other`` differ.

        Used on checkpoint load to warn the user that the current Objective's
        save configuration does not match the one that produced the checkpoint.
        """
        diffs: list[str] = []
        for f in self.to_dict():
            if getattr(self, f) != getattr(other, f):
                diffs.append(f)
        return diffs
