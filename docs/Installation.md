# Installation

## Prerequisites

- **Python ≥ 3.11**
- **Linux** (tested on Ubuntu 24.04 LTS and SUSE Linux Enterprise Server 15 SP6)
- (Optional) NVIDIA GPU with CUDA 12+ for accelerated simulations

## Install from PyPI

For normal use, install the published package with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install dfbench
```

The base install keeps heavyweight optimizer backends optional. Add extras for the algorithm families you need:

```bash
pip install "dfbench[optax,scipy]"      # common local optimizers
pip install "dfbench[evolution]"        # CMA, EvoX, Nevergrad, Evosax
pip install "dfbench[bo]"               # BoTorch/Ax/HEBO surrogate optimizers
pip install "dfbench[dfo,smac]"         # derivative-free and SMAC optimizers
pip install "dfbench[all]"              # all optimizer backends
pip install "dfbench[cuda13]"           # CUDA 13 JAX support
pip install "dfbench[cuda12]"           # CUDA 12 JAX support
pip install "dfbench[analysis]"         # notebooks, profiling, plotting helpers
```

`HEBO` is part of the `bo` and `all` extras. The optional backends are imported lazily: `import dfbench` succeeds even when an extra is not installed, and importing an algorithm whose backend is missing raises an `ImportError` pointing to the extra to install (e.g. `uv add 'dfbench[bo]'`).

## Development Install with `uv`

[uv](https://uv.dev/) is a fast Python package manager that handles virtual environments and dependency resolution automatically.

### CPU-only

```bash
uv sync
```

### With CUDA 12 (GPU support)

```bash
uv sync --group cuda12
```

### With analysis tools (profiling, notebooks)

```bash
uv sync --group analysis
```

### With publishing tools

```bash
uv sync --group publish
```

### Everything (GPU + analysis)

```bash
uv sync --group cuda12 --group analysis
```

## Development Install with `pip`

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Basic install
pip install -e .

# With CUDA GPU support
pip install -e ".[cuda12]"

# With all optimizer backends and analysis tools
pip install -e ".[all,analysis]"
```

## Verify Installation

```python
import dfbench
from dfbench.problems import VoyagerProblem

problem = VoyagerProblem()
print(f"Problem: {problem.name}, params: {problem.n_params}")
```

If this runs without error, you're ready to go.

## Dependencies

The project's core dependencies (managed in `pyproject.toml`) are:

| Package | Purpose |
|---------|---------|
| `differometor` | JAX-based interferometer simulator |
| `jax` / `jaxlib` | Auto-differentiation, JIT compilation, vmap |
| `numpy` | Numerical arrays |
| `jaxtyping` | Type annotations for JAX arrays |
| `matplotlib` | Plotting helpers for objectives and problems |

Optimizer backends such as `optax`, `scipy`, `torch`, `botorch`, `evox`, `nevergrad`, `OMADS`, `pdfo`, and `smac` are installed through the extras above. The `t2j` and `j2t` helpers import `torch` only when called, so `import dfbench` does not require PyTorch.

## Build and Publish Checks

Before uploading a release, build the distributions and validate the metadata:

```bash
source .venv/bin/activate
python -m pip install --upgrade build twine
rm -rf dist/
python -m build
python -m twine check dist/*
```

Install the built wheel in a clean environment before uploading:

```bash
python -m venv /tmp/dfbench-wheel-test
source /tmp/dfbench-wheel-test/bin/activate
python -m pip install dist/*.whl
python -c "import dfbench; from dfbench import Objective; print('dfbench import ok')"
```

Upload to TestPyPI first, then PyPI once the TestPyPI install succeeds:

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

## HPC Notes

On shared HPC file systems, matplotlib may fail because its default config directory is read-only. The framework handles this automatically by setting `MPLCONFIGDIR` to a temporary directory before any imports (see `core/_init_env.py`). No manual action is needed.

If you encounter permission-related errors on job submission, ensure the `data/` output directories are on a writable filesystem.
