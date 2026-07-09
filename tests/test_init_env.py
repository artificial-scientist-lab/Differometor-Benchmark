"""Section 11: Environment initialization.

Tests 11.1-11.2.
"""

from __future__ import annotations

import os
import importlib


class TestMplConfigDir:
    def test_set_after_import(self):
        """11.1 After import, MPLCONFIGDIR is set in os.environ."""
        import dfbench.core._init_env  # noqa: F401

        assert "MPLCONFIGDIR" in os.environ, "MPLCONFIGDIR should be set after import"
        assert len(os.environ["MPLCONFIGDIR"]) > 0

    def test_not_overwritten(self, monkeypatch):
        """11.2 If MPLCONFIGDIR was already set, it is not overwritten."""
        monkeypatch.setenv("MPLCONFIGDIR", "/my/custom/path")
        # Re-import the module to trigger the check
        importlib.reload(importlib.import_module("dfbench.core._init_env"))
        assert os.environ["MPLCONFIGDIR"] == "/my/custom/path"
