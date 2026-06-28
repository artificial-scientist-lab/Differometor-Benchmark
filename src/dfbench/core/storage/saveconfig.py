"""Declarative save configuration for :class:`Objective`.

A :class:`SaveConfig` records *what* histories an Objective records during
a run. It replaces the nine individual boolean constructor arguments with
two standard flags plus a declarative ``save`` list of string tokens for
advanced combinations. The config is embedded in :class:`RunMetadata` so a
checkpoint records which histories were active, preventing silent
mismatches on resume.

Valid ``save`` tokens
--------------------

| Token              | Effect                                                              |
|--------------------|---------------------------------------------------------------------|
| ``"grad"``         | Record gradient history (reduced to one entry per eval for batches)|
| ``"hessian"``      | Record Hessian history (reduced to one entry per eval for batches) |
| ``"eval_type"``    | Record per-eval type bitmask history                                |
| ``"batched_losses"``| Store full ``(batch,)`` loss vectors instead of batch min           |
| ``"batched_grads"``| Store full ``(batch, n_params)`` gradient arrays                    |
| ``"batched_hessians"``| Store full ``(batch, n_params, n_params)`` Hessian arrays       |
| ``"batched_params"``| Store full ``(batch, n_params)`` parameter arrays                  |
| ``"batched"``      | Convenience alias: expands to ``batched_params``, ``batched_losses``,|
|                    | ``batched_grads``, ``batched_hessians``                            |

The two standard flags (``save_time_steps``, ``save_params_history``)
remain as explicit booleans because they are the most commonly toggled and
have less combinatorial interaction with the batched variants.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid advanced save tokens.
VALID_TOKENS: frozenset[str] = frozenset(
    {
        "grad",
        "hessian",
        "eval_type",
        "batched_losses",
        "batched_grads",
        "batched_hessians",
        "batched_params",
        "batched",  # convenience alias, expanded at construction
    }
)

# Expansion of the "batched" convenience alias.
_BATCHED_EXPANSION: list[str] = [
    "batched_params",
    "batched_losses",
    "batched_grads",
    "batched_hessians",
]


@dataclass
class SaveConfig:
    """Declarative record of which histories an Objective tracks.

    Attributes:
        time_steps: Record elapsed-time timestamps per evaluation.
        params: Record parameter vectors (reduced for batches).
        grad: Record gradient vectors (reduced for batches).
        hessian: Record Hessian matrices (reduced for batches).
        eval_type: Record per-eval type bitmask history.
        batched_losses: Store full batched loss vectors.
        batched_grads: Store full batched gradient arrays.
        batched_hessians: Store full batched Hessian arrays.
        batched_params: Store full batched parameter arrays.
    """

    time_steps: bool = True
    params: bool = True
    grad: bool = False
    hessian: bool = False
    eval_type: bool = False
    batched_losses: bool = False
    batched_grads: bool = False
    batched_hessians: bool = False
    batched_params: bool = False

    @classmethod
    def from_flags(
        cls,
        save_time_steps: bool = True,
        save_params_history: bool = True,
        save: list[str] | None = None,
    ) -> "SaveConfig":
        """Build a :class:`SaveConfig` from the two standard flags + token list.

        Args:
            save_time_steps: Record timestamps.
            save_params_history: Record parameter history.
            save: List of advanced tokens (see module docstring).

        Raises:
            ValueError: If an unknown token is encountered.
        """
        cfg = cls(time_steps=save_time_steps, params=save_params_history)
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
                elif t == "batched_losses":
                    cfg.batched_losses = True
                elif t == "batched_grads":
                    cfg.batched_grads = True
                elif t == "batched_hessians":
                    cfg.batched_hessians = True
                elif t == "batched_params":
                    cfg.batched_params = True

        return cfg

    def to_dict(self) -> dict[str, bool]:
        """Serialize to a plain dict for embedding in :class:`RunMetadata`."""
        return {
            "time_steps": self.time_steps,
            "params": self.params,
            "grad": self.grad,
            "hessian": self.hessian,
            "eval_type": self.eval_type,
            "batched_losses": self.batched_losses,
            "batched_grads": self.batched_grads,
            "batched_hessians": self.batched_hessians,
            "batched_params": self.batched_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SaveConfig":
        """Reconstruct from a :meth:`to_dict` dict (missing keys → defaults)."""
        return cls(
            time_steps=bool(d.get("time_steps", True)),
            params=bool(d.get("params", True)),
            grad=bool(d.get("grad", False)),
            hessian=bool(d.get("hessian", False)),
            eval_type=bool(d.get("eval_type", False)),
            batched_losses=bool(d.get("batched_losses", False)),
            batched_grads=bool(d.get("batched_grads", False)),
            batched_hessians=bool(d.get("batched_hessians", False)),
            batched_params=bool(d.get("batched_params", False)),
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
