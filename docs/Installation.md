# Installation

## Prerequisites

- **Python ≥ 3.11**
- **Linux** (tested on Ubuntu 24.04 LTS and SUSE Linux Enterprise Server 15 SP6)
- (Optional) NVIDIA GPU with CUDA 12+ for accelerated simulations

## Recommended: Install with `uv`

[uv](https://uv.dev/) is a fast Python package manager that handles virtual environments and dependency resolution automatically.

### CPU-only

```bash
uv sync
```

### With CUDA 12 (GPU support)

```bash
uv sync --group cuda12
```

### With development tools (profiling, notebooks, testing)

```bash
uv sync --group dev
```

### Everything (GPU + dev)

```bash
uv sync --group cuda12 --group dev
```

## Alternative: Install with `pip`

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Basic install
pip install -e .

# With CUDA GPU support
pip install -e .
pip install --upgrade "jax[cuda12]==0.9.0.1"
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
| `jaxtyping` | Type annotations for JAX arrays |
| `botorch` | Bayesian optimisation (BO and TuRBO algorithms) |
| `evox` | Evolutionary optimisation (PSO, CMA-ES, etc.) |
| `beartype` | Runtime type checking |

## HPC Notes

On shared HPC file systems, matplotlib may fail because its default config directory is read-only. The framework handles this automatically by setting `MPLCONFIGDIR` to a temporary directory before any imports (see `core/_init_env.py`). No manual action is needed.

If you encounter permission-related errors on job submission, ensure the `data/` output directories are on a writable filesystem.
