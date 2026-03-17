"""Live in-place CLI display for Objective optimisation progress.

When ``display_mode="live"`` (the default) the :class:`LiveDisplay` renderer
overwrites the same block of terminal lines on every update, producing a
compact, continuously-refreshing dashboard:

.. code-block:: text

    ┌─ MyAlgorithm × MyProblem (12 params) ──────────────────────────────────────┐
    │ Time   [██████████████████░░░░░░░░░░] 60.0%  60.0s / 100.0s   ETA  40.0s  │
    │ Evals  [████████░░░░░░░░░░░░░░░░░░░░] 30.0%  1,500 / 5,000    ETA  2m 4s  │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ Best Loss      3.412e-03    Current Loss   8.764e-03                        │
    │ Improvements   14           Since Improv.  86                               │
    │ Evals/sec      32.3         Avg Batch      8.0                              │
    │ Loss Trend   ▇▆▅▄▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁   Δ -3.41e-02                           │
    │ Call Types   val+g 96%  val 4%                                              │
    │ Checkpoint   last @ 1,200  ·  next in 48 evals  (every 500)                │
    │ Device Mem   1.23 GB  peak 1.45 GB  (61%)                                  │
    └────────────────────────────────────────────────────────────────────────────┘

When ``display_mode="log"`` the original multi-line log blocks are printed
periodically (same behaviour as the legacy ``verbose >= 1`` path).
"""

from __future__ import annotations

import sys
import shutil
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfbench.core.objective import Objective


# ── ANSI helpers ──────────────────────────────────────────────────────

_MOVE_UP = "\033[{}A"  # parameterised: \033[<n>A
_CLEAR_LINE = "\033[2K"


def _supports_ansi() -> bool:
    """Heuristic: does the current stdout support ANSI escape codes?"""
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


# ── Text helpers ──────────────────────────────────────────────────────

#: Unicode block characters used for sparklines (▁ = low, █ = high)
_SPARK = "▁▂▃▄▅▆▇█"
#: Unicode block characters used for progress bars
_BAR_FULL = "█"
_BAR_EMPTY = "░"

#: Human-readable names for eval-type bitmask codes
_TYPE_NAMES: dict[int, str] = {
    1: "val",
    2: "grad",
    3: "val+g",
    8: "hess",
    9: "val+h",
    10: "g+h",
    11: "val+g+h",
    5: "vmap-v",
    6: "vmap-g",
    7: "vmap+g",
    12: "vmap-h",
    13: "vmap-v+h",
    14: "vmap-g+h",
    15: "vmap+g+h",
    -1: "other",
}


def _bar(fraction: float, width: int = 30) -> str:
    """Return a text progress bar for *fraction* ∈ [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    n = int(round(fraction * width))
    return _BAR_FULL * n + _BAR_EMPTY * (width - n)


def _fmt_time(seconds: float) -> str:
    """Format seconds as ``Xh Ym Zs`` (or shorter for small values)."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def _fmt_loss(n: int | float | None) -> str:
    """Format a loss value: scientific notation for small values, else fixed."""
    if n is None:
        return "—"
    if isinstance(n, float):
        if n != n:  # NaN
            return "NaN"
        if n == float("inf"):
            return "∞"
        if n == float("-inf"):
            return "-∞"
        if abs(n) < 1e-2 and n != 0.0:
            return f"{n:.6e}"
        return f"{n:.6f}"
    return f"{n:,}"


def _fmt_bytes(b: int) -> str:
    """Format a byte count as a human-readable string (B / KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b //= 1024
    return f"{b:.2f} TB"


def _get_jax_memory() -> tuple[int | None, int | None, int | None]:
    """Query JAX device 0 memory statistics without side-effects.

    Returns
    -------
    (bytes_in_use, peak_bytes_in_use, bytes_limit)
        Each element is ``None`` when the backend does not support
        ``memory_stats()`` (e.g. CPU) or when the call fails.
    """
    try:
        import jax

        device = jax.devices()[0]
        stats = device.memory_stats()
        if stats is None:
            return None, None, None
        return (
            stats.get("bytes_in_use"),
            stats.get("peak_bytes_in_use"),
            stats.get("bytes_limit"),
        )
    except Exception:
        return None, None, None


def _sparkline(values: list[float], width: int) -> str:
    """Build a Unicode sparkline of exactly *width* characters.

    Lower values map to ``▁``, higher to ``█``.  Since loss optimisation
    aims to decrease the loss, a healthy run shows a left-to-right
    downward trend (``▇▅▄▃▂▁▁▁``).

    NaN entries are rendered as ``·``.
    """
    if not values or width <= 0:
        return "─" * max(0, width)

    # Sub-sample to at most `width` evenly-spaced points
    n = len(values)
    if n > width:
        indices = [int(i * n / width) for i in range(width)]
        values = [values[i] for i in indices]

    finite = [v for v in values if v == v]  # exclude NaN
    if not finite:
        return "·" * len(values)

    vmin, vmax = min(finite), max(finite)
    n_levels = len(_SPARK) - 1

    chars = []
    for v in values:
        if v != v:  # NaN
            chars.append("·")
        elif vmax == vmin:
            chars.append(_SPARK[0])
        else:
            idx = int((v - vmin) / (vmax - vmin) * n_levels)
            chars.append(_SPARK[min(idx, n_levels)])
    return "".join(chars)


# ── Live display class ───────────────────────────────────────────────


class LiveDisplay:
    """Renders a continuously-updated optimisation dashboard in the terminal.

    The display overwrites its own output on each :meth:`render` call using
    ANSI escape codes so that only a fixed block of lines is ever visible.

    Parameters
    ----------
    objective : Objective
        The ``Objective`` instance whose state is displayed.
    """

    def __init__(self, objective: "Objective") -> None:
        self._obj = objective
        self._use_ansi = _supports_ansi()
        self._rendered_once = False
        self._frame_lines = 0
        self._first_eval_time: float | None = None

    # ── public API ────────────────────────────────────────────────

    def render(self) -> None:
        """Build and print (or overwrite) the live dashboard."""
        lines = self._build_lines()

        # Move cursor up to overwrite previous frame
        if self._rendered_once and self._use_ansi and self._frame_lines > 0:
            sys.stdout.write(_MOVE_UP.format(self._frame_lines))

        for line in lines:
            if self._use_ansi:
                sys.stdout.write(_CLEAR_LINE + line + "\n")
            else:
                sys.stdout.write(line + "\n")

        sys.stdout.flush()
        self._frame_lines = len(lines)
        self._rendered_once = True

    def finalize(self) -> None:
        """Print a final, non-overwritable summary after the run."""
        lines = self._build_lines(final=True)
        if self._rendered_once and self._use_ansi and self._frame_lines > 0:
            sys.stdout.write(_MOVE_UP.format(self._frame_lines))
        for line in lines:
            if self._use_ansi:
                sys.stdout.write(_CLEAR_LINE + line + "\n")
            else:
                sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self._rendered_once = False  # don't overwrite the final summary

    # ── internal ──────────────────────────────────────────────────

    def _terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80

    def _build_lines(self, final: bool = False) -> list[str]:  # noqa: C901
        obj = self._obj
        tw = max(self._terminal_width(), 52)
        inner = tw - 2  # chars between the two │ border characters

        # ── Stat gathering ────────────────────────────────────────
        algorithm = obj.algorithm_str or "unknown"
        problem = obj.problem.name if hasattr(obj.problem, "name") else "problem"
        n_params = obj.n_params if obj.bounds is not None else "?"
        eval_count = obj.eval_count
        max_evals = obj._max_evals
        max_time = obj._max_time
        time_elapsed = obj.time_elapsed
        best_loss = obj.best_loss
        improvement_count = obj.improvement_count
        evals_since_imp = obj.evals_since_improvement
        log_calls = obj.log_call_count
        type_counts = obj.eval_type_counts
        ckpt_every = obj._save_to_file_every
        last_ckpt = obj.last_checkpoint_eval

        # Track first-eval wall time for throughput
        if eval_count > 0 and self._first_eval_time is None:
            self._first_eval_time = time.time()

        # Current loss scalar
        current_loss: float | None = None
        try:
            import jax.numpy as jnp

            raw = obj.current_loss
            if raw is not None:
                nd = getattr(raw, "ndim", 0)
                current_loss = float(raw) if nd == 0 else float(jnp.nanmin(raw))
        except Exception:
            pass

        # Evals per second (from first-logged eval to now)
        evals_per_sec = 0.0
        if self._first_eval_time is not None and eval_count > 0:
            dt = time.time() - self._first_eval_time
            if dt > 0.05:
                evals_per_sec = eval_count / dt

        # ── Row helpers ───────────────────────────────────────────
        def _pad(content: str) -> str:
            """Pad/truncate content to exactly `inner` chars."""
            n = len(content)
            if n >= inner:
                return content[:inner]
            return content + " " * (inner - n)

        def _row(content: str) -> str:
            return f"│{_pad(content)}│"

        def _sep() -> str:
            return f"├{'─' * inner}┤"

        def _row2(lbl1: str, val1: str, lbl2: str, val2: str) -> str:
            """Two-stat row with fixed half-width columns."""
            half = inner // 2
            left = f" {lbl1:<13}{val1}"
            right = f" {lbl2:<13}{val2}"
            # Pad left column to half, then append right
            lpad = " " * max(0, half - len(left))
            content = left + lpad + right
            return _row(content)

        # ETA helper
        def _eta(remaining_evals=None, remaining_time=None) -> str:
            candidates: list[float] = []
            if remaining_time is not None and remaining_time >= 0:
                candidates.append(remaining_time)
            if remaining_evals is not None and evals_per_sec > 0:
                candidates.append(remaining_evals / evals_per_sec)
            if not candidates:
                return ""
            return f"  ETA {_fmt_time(min(candidates))}"

        # ── Title ─────────────────────────────────────────────────
        title = f" {algorithm} × {problem} ({n_params} params) "
        if final:
            title = f" {algorithm} × {problem} — DONE "
        fill = "─" * max(0, inner - 1 - len(title))
        lines: list[str] = [f"┌─{title}{fill}┐"]

        # ── Progress bars ─────────────────────────────────────────
        # Reserve chars: " Time   [" (8) + "] pct  elapsed / max  ETA XXs" (~38)
        # → bar occupies the rest
        bar_width = max(10, tw - 54)

        if max_time is not None:
            t_frac = min(1.0, time_elapsed / max_time)
            time_left = max(0.0, max_time - time_elapsed)
            eta_t = _eta(remaining_time=time_left)
            row_t = (
                f" Time   [{_bar(t_frac, bar_width)}]"
                f" {t_frac * 100:5.1f}%  {_fmt_time(time_elapsed)} / {_fmt_time(max_time)}"
                f"{eta_t}"
            )
        else:
            row_t = f" Time   {_fmt_time(time_elapsed)}  (no time limit)"
        lines.append(_row(row_t))

        if max_evals is not None and max_evals > 0:
            e_frac = min(1.0, eval_count / max_evals)
            evals_left = max(0, max_evals - eval_count)
            eta_e = _eta(remaining_evals=evals_left)
            row_e = (
                f" Evals  [{_bar(e_frac, bar_width)}]"
                f" {e_frac * 100:5.1f}%  {eval_count:,} / {max_evals:,}"
                f"{eta_e}"
            )
        else:
            row_e = f" Evals  {eval_count:,}  (no eval limit)"
        lines.append(_row(row_e))

        lines.append(_sep())

        # ── Stats block ───────────────────────────────────────────
        lines.append(
            _row2(
                "Best Loss",
                _fmt_loss(best_loss),
                "Current Loss",
                _fmt_loss(current_loss),
            )
        )

        avg_batch = (eval_count / log_calls) if log_calls > 0 else None
        eps_str = f"{evals_per_sec:,.1f}" if evals_per_sec > 0 else "—"
        batch_str = f"{avg_batch:.1f}" if avg_batch is not None else "—"
        lines.append(_row2("Evals/sec", eps_str, "Avg Batch", batch_str))

        lines.append(
            _row2(
                "Improvements",
                str(improvement_count),
                "Since Improv.",
                str(evals_since_imp),
            )
        )

        # ── Loss trend sparkline ──────────────────────────────────
        try:
            losses = obj.loss_history_reduced
        except Exception:
            losses = []

        # Deduplicate consecutive equal values (running-minimum trajectory)
        running_min: list[float] = []
        cur_min = float("inf")
        for v in losses:
            if v < cur_min:
                cur_min = v
            running_min.append(cur_min)

        if running_min:
            # Sparkline occupies the space remaining after label + Δbest annotation
            # " Loss Trend   " (14) + spark + "  Δ " (4) + "+1.23e-04" (9) = 27 + spark
            spark_width = max(8, inner - 29)
            spark = _sparkline(running_min, spark_width)
            # delta = improvement over the window shown in the sparkline
            window = min(len(running_min), spark_width)
            delta = running_min[-1] - running_min[-window]
            delta_str = f"{delta:+.2e}"
            lines.append(_row(f" Loss Trend   {spark}  Δ {delta_str}"))
        else:
            lines.append(_row(" Loss Trend   (no data yet)"))

        # ── Call types ────────────────────────────────────────────
        if type_counts:
            total = sum(type_counts.values())
            parts = [
                f"{_TYPE_NAMES.get(code, f'#{code}')} {100 * cnt // total}%"
                for code, cnt in sorted(type_counts.items(), key=lambda kv: -kv[1])
            ]
            lines.append(_row(f" Call Types   {'  '.join(parts)}"))

        # ── Checkpoint status ─────────────────────────────────────
        if ckpt_every is not None:
            if last_ckpt is not None:
                evals_since_ckpt = eval_count - last_ckpt
                next_in = ckpt_every - evals_since_ckpt
                ckpt_str = (
                    f"last @ {last_ckpt:,}  ·  next in {next_in:,} evals"
                    f"  (every {ckpt_every:,})"
                )
            else:
                next_in = ckpt_every - (eval_count % ckpt_every if ckpt_every else 0)
                ckpt_str = (
                    f"next in {next_in:,} evals  (every {ckpt_every:,}, none yet)"
                )
            lines.append(_row(f" Checkpoint   {ckpt_str}"))

        # ── JAX device memory ─────────────────────────────────────
        mem_used, mem_peak, mem_limit = _get_jax_memory()
        if mem_used is not None:
            mem_str = _fmt_bytes(mem_used)
            if mem_peak is not None:
                mem_str += f"  peak {_fmt_bytes(mem_peak)}"
            if mem_limit is not None and mem_limit > 0:
                mem_str += f"  ({100 * mem_used // mem_limit}%)"
            lines.append(_row(f" Device Mem   {mem_str}"))
        lines.append(f"└{'─' * inner}┘")
        return lines


# ── Log-style display class ───────────────────────────────────────────


class LogDisplay:
    """Renders periodic scrolling log blocks.

    Falls back to this when ``display_mode="log"`` or stdout is not a TTY.
    Unlike :class:`LiveDisplay`, each call appends new lines to the terminal
    rather than overwriting the previous frame.
    """

    def __init__(self, objective: "Objective") -> None:
        self._obj = objective

    def render(self) -> None:
        """Print a log block with current stats."""
        obj = self._obj
        s = obj.get_summary()
        log_calls = obj.log_call_count
        type_counts = obj.eval_type_counts
        ckpt_every = obj._save_to_file_every
        last_ckpt = obj.last_checkpoint_eval
        avg_batch = (s["eval_count"] / log_calls) if log_calls > 0 else None

        parts = [
            "───────────────",
            f"evals    = {s['eval_count']}",
            f"best     = {s['best_loss']}",
            f"current  = {s['current_loss']}",
            f"time     = {s['time_elapsed']:.2f}s",
            f"improv   = {s['improvement_count']}",
            f"since    = {s['evals_since_improvement']}",
        ]
        if avg_batch is not None:
            parts.append(f"batch    = {avg_batch:.1f}")
        if type_counts:
            total = sum(type_counts.values())
            type_str = "  ".join(
                f"{_TYPE_NAMES.get(c, f'#{c}')} {100 * n // total}%"
                for c, n in sorted(type_counts.items(), key=lambda kv: -kv[1])
            )
            parts.append(f"types    = {type_str}")
        if ckpt_every is not None:
            if last_ckpt is not None:
                parts.append(f"ckpt     = last @ {last_ckpt:,}")
            else:
                parts.append(f"ckpt     = none yet (every {ckpt_every:,})")

        try:
            print("\n".join(parts))
        except Exception:
            pass

    def finalize(self) -> None:
        """Print a final summary (identical to a regular render in log mode)."""
        self.render()
