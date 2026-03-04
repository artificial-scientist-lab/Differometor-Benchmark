"""Section 10 — Display helper functions.

Tests 10.1–10.7.
"""

from __future__ import annotations

import math

import pytest

from dfbench.core.display import (
    LiveDisplay,
    LogDisplay,
    _bar,
    _fmt_loss,
    _fmt_time,
    _sparkline,
)
from dfbench.core.objective import Objective


# ======================================================================
# _bar (10.1–10.2)
# ======================================================================


class TestBar:
    def test_zero(self):
        """10.1 _bar(0.0) returns all-empty chars."""
        result = _bar(0.0, width=10)
        assert "█" not in result
        assert len(result) == 10

    def test_one(self):
        """10.1b _bar(1.0) returns all-full chars."""
        result = _bar(1.0, width=10)
        assert "░" not in result
        assert len(result) == 10

    def test_clamp_above(self):
        """10.2 Values > 1 are clamped."""
        result = _bar(1.5, width=10)
        assert "░" not in result

    def test_clamp_below(self):
        """10.2b Values < 0 are clamped."""
        result = _bar(-0.5, width=10)
        assert "█" not in result


# ======================================================================
# _fmt_time (10.3)
# ======================================================================


class TestFmtTime:
    def test_seconds(self):
        """10.3 Formats small values as seconds."""
        assert "s" in _fmt_time(5.3)
        assert "m" not in _fmt_time(5.3)

    def test_minutes(self):
        """10.3b Minutes for 60-3600s."""
        result = _fmt_time(125)
        assert "m" in result

    def test_hours(self):
        """10.3c Hours for >= 3600s."""
        result = _fmt_time(7200)
        assert "h" in result

    def test_negative(self):
        """10.3d Negative → '0s'."""
        assert _fmt_time(-5) == "0s"


# ======================================================================
# _fmt_loss (10.4)
# ======================================================================


class TestFmtLoss:
    def test_none(self):
        """10.4 None → '—'."""
        assert _fmt_loss(None) == "—"

    def test_nan(self):
        """10.4b NaN → 'NaN'."""
        assert _fmt_loss(float("nan")) == "NaN"

    def test_inf(self):
        """10.4c inf → '∞'."""
        assert _fmt_loss(float("inf")) == "∞"

    def test_neg_inf(self):
        """10.4d -inf → '-∞'."""
        assert _fmt_loss(float("-inf")) == "-∞"

    def test_small_float(self):
        """10.4e Small floats → scientific notation."""
        result = _fmt_loss(0.00123)
        assert "e" in result

    def test_integer(self):
        """10.4f Integers formatted with comma separator."""
        result = _fmt_loss(1000000)
        assert "," in result


# ======================================================================
# _sparkline (10.5)
# ======================================================================


class TestSparkline:
    def test_correct_width(self):
        """10.5 Output has the requested width."""
        result = _sparkline([1.0, 2.0, 3.0, 4.0], width=4)
        assert len(result) == 4

    def test_empty(self):
        """10.5b Handles empty input."""
        result = _sparkline([], width=5)
        assert len(result) == 5

    def test_all_nan(self):
        """10.5c All-NaN → dots."""
        result = _sparkline([float("nan"), float("nan")], width=2)
        assert "·" in result

    def test_subsampling(self):
        """10.5d When values > width, subsamples."""
        result = _sparkline(list(range(100)), width=10)
        assert len(result) == 10


# ======================================================================
# LiveDisplay & LogDisplay (10.6–10.7)
# ======================================================================


class TestLiveDisplay:
    def test_render_no_crash(self, mock_problem, capsys):
        """10.6 LiveDisplay.render does not crash on a fresh Objective."""
        obj = Objective(mock_problem)
        display = LiveDisplay(obj)
        display.render()  # Should not raise


class TestLogDisplay:
    def test_render_no_crash(self, mock_problem, capsys):
        """10.7 LogDisplay.render does not crash on a fresh Objective."""
        obj = Objective(mock_problem)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        display = LogDisplay(obj)
        display.render()  # Should not raise
