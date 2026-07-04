"""Environment initialization for the optimization package.

This module is imported before any other modules to configure environment
variables that must be set before importing certain libraries (e.g., matplotlib).

This module should be imported at the very top of optimization/__init__.py.
"""

import os
import tempfile

# Set matplotlib config directory to avoid permission issues on HPC systems.
# This must be set BEFORE matplotlib is imported anywhere in the codebase.
# Using a temp directory avoids write permission issues on shared filesystems.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mpl_config_")

# Force a non-interactive backend so plotting works in headless / CI
# environments. On some Windows runners (e.g. GitHub Actions windows-latest
# with Python 3.12) the default TkAgg backend fails because the bundled Tcl
# install is incomplete, raising `_tkinter.TclError: couldn't read file
# ".../tcl/tcl8.6/auto.tcl"`. Setting MPLBACKEND=Agg (when not overridden by
# the user) avoids the Tk fallback entirely.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
