"""Human-readable export of a :class:`RunState` (JSON + PNG plots).

This module replaces the old :meth:`Objective.output_to_files` by
treating the human-readable artifacts as a *derived view* over the
canonical :class:`RunState`, not as a second write path inside the
Objective. Plotting is split into pure functions that return matplotlib
figures, and writing those figures/JSON to disk is a separate step that
goes through a :class:`StorageBackend`-compatible interface (here the
local filesystem, but trivially redirectable).

For optical problems that expose ``calculate_sensitivity`` /
``_frequencies`` / ``_target_sensitivities``, a sensitivity plot is
produced in addition to the loss curve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np

from dfbench.core.storage.state import RunState


@runtime_checkable
class ProblemView(Protocol):
    """Subset of a problem needed for export, used for typing only."""

    name: str
    _frequencies: np.ndarray
    _target_sensitivities: np.ndarray

    def calculate_sensitivity(self, params) -> np.ndarray: ...


# ------------------------------------------------------------------
# Pure plotting functions (return figures, do not touch the filesystem)
# ------------------------------------------------------------------


def plot_loss_curve(losses: np.ndarray) -> plt.Figure:
    """Return a matplotlib figure of the loss history."""
    fig, ax = plt.subplots()
    ax.plot(np.asarray(losses))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.axhline(0, color="red", linestyle="--")
    ax.grid()
    fig.tight_layout()
    return fig


def plot_sensitivity(
    frequencies: np.ndarray,
    sensitivities: np.ndarray,
    target: np.ndarray | None = None,
) -> plt.Figure:
    """Return a log-log sensitivity figure for optical problems."""
    fig, ax = plt.subplots()
    ax.plot(frequencies, sensitivities, label="Optimized Sensitivity")
    if target is not None:
        ax.plot(frequencies, target, label="Target Sensitivity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sensitivity [/sqrt(Hz)]")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Exporter
# ------------------------------------------------------------------


def _has_sensitivity(problem: Any) -> bool:
    return (
        problem is not None
        and hasattr(problem, "calculate_sensitivity")
        and hasattr(problem, "_frequencies")
    )


def _to_json_list(arr: np.ndarray) -> list:
    """Convert an object-dtype history array to a JSON-serialisable list.

    Each element may be a JAX scalar, JAX array, numpy scalar, numpy array,
    or None.  ``np.asarray(arr, dtype=object).tolist()`` alone preserves the
    original objects, which ``json.dump`` cannot serialise, so each element
    is explicitly converted via ``np.asarray(x).tolist()``.
    """
    a = np.asarray(arr, dtype=object)
    if a.size == 0:
        return []
    return [np.asarray(x).tolist() if x is not None else None for x in a.tolist()]


class RunDataExporter:
    """Write human-readable JSON + PNG summaries for a :class:`RunState`.

    Kept deliberately small: it derives everything from a
    :class:`RunState` (which the Objective produces) plus the optional
    underlying problem (for sensitivity plots) and a destination
    directory. It does not read or mutate the Objective.
    """

    def __init__(self, root: str | Path = "./data/problem_output") -> None:
        """Initialize the exporter.

        Args:
            root: Base directory for human-readable outputs. Defaults to
                ``./data/problem_output`` to match the historical layout.
        """
        self.root = Path(root)

    def output_dir(
        self,
        problem_name: str,
        algorithm_name: str,
        hyper_param_str: str | None = None,
    ) -> Path:
        """Return (and create) the output directory for a run."""
        d = self.root / Path(problem_name).name
        algo = algorithm_name.strip("_") if algorithm_name else ""
        hp = hyper_param_str.strip("_") if hyper_param_str else ""
        if algo and hp:
            d = d / f"{algo}_{hp}"
        elif algo:
            d = d / algo
        d.mkdir(parents=True, exist_ok=True)
        return d

    def export(
        self,
        state: RunState,
        problem: Any | None = None,
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
        print_summary: bool = True,
        write_parameters_json: bool = True,
        write_losses_json: bool = True,
        write_losses_png: bool = True,
        write_sensitivity_png: bool = True,
    ) -> Path:
        """Write JSON + PNG artifacts for ``state`` and return the dir.

        Each artifact is independently optional; pass ``write_*`` as
        ``False`` to skip that file. The output directory is still created
        and returned regardless of which artifacts are written.

        Args:
            state: The run snapshot to export.
            problem: Optional underlying problem; used for the problem
                name and, if it exposes sensitivity data, for an extra
                sensitivity plot.
            hyper_param_str: Hyperparameter string for directory/file naming.
            hyper_param_str_in_filename: Include the hyperparameter string
                in the filename suffix.
            print_summary: Print best params/loss and the output dir.
            write_parameters_json: Write the best-parameters JSON file.
            write_losses_json: Write the loss-history JSON file.
            write_losses_png: Write the loss-curve PNG plot.
            write_sensitivity_png: Write the sensitivity PNG plot. Only
                produced when the problem exposes sensitivity data and
                ``best_params`` is non-empty; the flag is an additional
                gate on top of that condition.
        """
        problem_name = (
            getattr(problem, "name", None) or state.metadata.problem_name or "problem"
        )
        algorithm_name = state.metadata.algorithm_name or "unknown"
        if not hyper_param_str:
            hyper_param_str = state.metadata.hyper_param_str or ""

        best_params = (
            np.asarray(state.best_params) if state.best_params.size > 0 else None
        )
        losses = _to_json_list(state.loss_history)

        if print_summary:
            print(f"Parameters of the best solution: {best_params}")
            print(f"Best loss: {state.best_loss}")

        algo_fmt = f"_{algorithm_name.strip('_')}" if algorithm_name else ""
        hp_fmt = f"_{hyper_param_str.strip('_')}" if hyper_param_str else ""
        timestamp = state.metadata.timestamp
        prefix = f"{problem_name}{algo_fmt}_{timestamp}"
        suffix = hp_fmt if hyper_param_str_in_filename else ""

        out_dir = self.output_dir(problem_name, algorithm_name, hyper_param_str)
        if print_summary:
            print(f"Output directory: {out_dir}")

        # JSON: parameters
        if write_parameters_json:
            with open(out_dir / f"{prefix}_parameters{suffix}.json", "w") as f:
                json.dump(
                    best_params.tolist() if best_params is not None else None,
                    f,
                    indent=4,
                )

        # JSON: losses
        if write_losses_json:
            with open(out_dir / f"{prefix}_losses{suffix}.json", "w") as f:
                json.dump(losses, f, indent=4)

        # PNG: loss curve
        if write_losses_png:
            fig = plot_loss_curve(np.asarray(state.loss_history, dtype=object))
            fig.savefig(out_dir / f"{prefix}_losses{suffix}.png")
            plt.close(fig)

        # PNG: sensitivity (optical problems only)
        if (
            write_sensitivity_png
            and _has_sensitivity(problem)
            and best_params is not None
        ):
            sens_problem: ProblemView = problem  # type: ignore[assignment]
            sensitivities = sens_problem.calculate_sensitivity(best_params)
            target = getattr(sens_problem, "_target_sensitivities", None)
            fig = plot_sensitivity(
                np.asarray(sens_problem._frequencies),
                np.asarray(sensitivities),
                np.asarray(target) if target is not None else None,
            )
            fig.savefig(out_dir / f"{prefix}_sensitivity{suffix}.png")
            plt.close(fig)

        return out_dir
