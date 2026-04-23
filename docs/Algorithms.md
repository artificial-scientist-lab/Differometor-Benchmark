# Algorithms

All built-in algorithms subclass `OptimizationAlgorithm` and follow a common contract: they receive a pre-configured `Objective`, run their optimization loop, and mutate it in place (logging all results). `optimize()` returns `None` — the caller accesses results from the same `Objective` instance it passed in.

**Import:**

```python
from dfbench.algorithms import (
    AdamGD, SAGD, NAAdamGD, LBFGSGD,       # gradient-based (original)
    OptaxAdam, OptaxAdamW, OptaxSAM,        # gradient-based (Optax batch, 34 total)
    BFGS, LBFGSB, NonlinearCG, NewtonCG,   # SciPy gradient / quasi-Newton
    TrustNCG, TrustKrylov, TrustConstr,    # SciPy trust-region / constrained
    TNC, SLSQP, COBYQA, COBYLA, Dogleg, SR1,
    RandomSearch, EvoxPSO, EvoxES,           # evolutionary
    NevergradOnePlusOne, NevergradTBPSA,     # nevergrad baselines
    NevergradNGOpt,                          # nevergrad meta-optimizer
    OmadsMADS, OmadsOrthoMADS,               # derivative-free direct search
    BotorchBO, BotorchTuRBO,                 # surrogate-based
    VAESampling,                              # generative
)
```

Native-JAX custom/hybrid algorithms are also available:

```python
from dfbench.algorithms import (
    SGLDJAX, ASAMJAX, AdamToLBFGSJAX, EntropySGDJAX, SGHMCJAX,
    OGDJAX, OAdamJAX, PerturbedGDJAX, NoisyAdamJAX,
    GDRestartsJAX, GaussianSmoothingGDJAX, ARCJAX,
)
```

---

## Algorithm Types

Every algorithm declares an `algorithm_type` from the `AlgorithmType` enum:

| Type | `unbounded` | Evaluation methods used | Examples |
|------|-------------|------------------------|----------|
| `GRADIENT_BASED` | `True` | `value_and_grad()` | Adam, SA-GD, NA-Adam, L-BFGS, 34 Optax optimizers |
| `EVOLUTIONARY` | `False` | `vmap_value()` | Random Search, PSO, CMA-ES |
| `EVOLUTIONARY` (direct search) | `False` | `value()` | MADS, OrthoMADS |
| `SURROGATE_BASED` | `False` | `value()`, `vmap_value()` | Bayesian Optimization, TuRBO, ReSTIR |
| `GENERATIVE` | varies | `value()`, `vmap_value()` | VAE Sampling |

The `Benchmark` harness uses `algorithm_type` to set `unbounded` automatically. When running algorithms standalone, you set it yourself.

---

## Gradient-Based Algorithms

These algorithms use gradient information for optimization. Most are configured to work in unbounded $(-\infty, +\infty)$ space via sigmoid-transformed objectives, though some can work directly in bounded space depending on their implementation.

### AdamGD

Standard Adam optimizer with gradient clipping.

```python
optimizer = AdamGD()
optimizer.optimize(
    problem_objective=obj,
    learning_rate=0.1,      # Adam learning rate
    patience=1000,           # stop after N iters without improvement
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `learning_rate` | `0.1` | Learning rate for Adam. |
| `patience` | `1000` | Early stopping: halt after this many iterations without a new best loss. |

**Implementation detail:** Uses `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))`. The gradient clipping prevents exploding updates in the early phase of optimization.

---

### SAGD (Simulated Annealing Gradient Descent)

Based on [arXiv:2107.07558](https://arxiv.org/abs/2107.07558). Combines gradient descent with a simulated-annealing-style probabilistic gradient ascent to escape local minima.

```python
optimizer = SAGD()
optimizer.optimize(
    problem_objective=obj,
    learning_rate=0.1,
    patience=1000,
    T0=15.0,                  # initial temperature
    sigma=1.0,                # gradient ascent step expansion
    max_ascent_prob=0.33,     # cap on ascent probability
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `T0` | `15.0` | Initial temperature. Higher → more frequent gradient ascent. |
| `sigma` | `1.0` | Multiplicative expansion of the gradient when performing ascent. |
| `max_ascent_prob` | `0.33` | Hard cap on ascent probability. The paper recommends < 0.33 for convergence. |
| `use_double_annealing` | `False` | Use the "double SA" formula for exponentially decaying learning rates. |

**Rationale — why gradient ascent?** Local minima are a major issue in high-dimensional non-convex landscapes. SA-GD occasionally moves uphill with a probability that depends on the temperature and the loss difference, similar to Metropolis–Hastings. This gives the optimizer a chance to escape shallow local minima early in the run, while converging normally once the temperature cools.

---

### NAAdamGD (Noisy-Annealing Adam)

Adam with decaying Gaussian noise injection for exploration.

```python
optimizer = NAAdamGD()
optimizer.optimize(
    problem_objective=obj,
    learning_rate=0.1,
    patience=1000,
    noise_std_start=0.3,      # initial noise σ
    noise_std_end=0.0,        # final noise σ
    noise_schedule="exponential",
    noise_anneal_iters=5000,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `noise_std_start` | `0.3` | Initial noise standard deviation. |
| `noise_std_end` | `0.0` | Final noise standard deviation. |
| `noise_schedule` | `"exponential"` | Decay curve: `"linear"` or `"exponential"` (geometric interpolation). |
| `noise_injection` | `"update"` | Where noise is added: `"update"` (to Adam step) or `"params"` (to parameters directly). |
| `noise_clip_norm` | `None` | Hard cap on noise vector L2 norm. |
| `noise_anneal_iters` | `5000` | Iterations over which noise decays. Only used when `noise_anneal_budget_fraction` is not set. |
| `noise_anneal_budget_fraction` | `None` | If set, noise decays over this fraction of the total budget (via `budget_progress_fraction`). E.g. `0.5` means noise reaches `noise_std_end` at 50% of the budget. Takes priority over `noise_anneal_iters`. |
| `noise_cap_relative_to_update` | `0.25` | Caps noise to this fraction of the Adam update norm. |
| `noise_cap_start_iter` | `500` | Iteration at which relative capping activates. |

**Rationale — noise capping:** Without capping, noise can overwhelm the optimizer update when gradients are very small (near a plateau). The relative cap ensures noise never exceeds 25% (by default) of the Adam step magnitude.

---

### LBFGSGD

L-BFGS optimizer from Optax. Uses second-order curvature information for faster convergence on smooth landscapes.

> **Note:** Because `optax.lbfgs` needs the raw value function for its internal line-search, this algorithm JIT-compiles the full optimization step and uses `obj.log_evaluation()` to record results after each step (instead of calling `obj.value_and_grad()` directly). This makes it a useful reference for implementing other algorithms that require custom JIT-compiled evaluation loops — see `src/dfbench/algorithms/gradient_based/misc/lbfgs_gd.py`.

```python
optimizer = LBFGSGD()
optimizer.optimize(
    problem_objective=obj,
    patience=500,
    random_seed=42,
)
```

### SciPy Gradient / Trust / Constrained Family

The SciPy-backed optimizers follow the same `Objective` contract as the Optax-based ones while using `scipy.optimize.minimize` under the hood. Public classes include:

- `BFGS`, `LBFGSB`, `NonlinearCG`, `NewtonCG`
- `TrustNCG`, `TrustKrylov`, `TrustConstr`, `Dogleg`, `SR1`
- `TNC`, `SLSQP`, `COBYQA`, `COBYLA`

Bounded-vs-unbounded behavior is explicit in each class:

- Unbounded sigmoid-space defaults: `BFGS`, `NonlinearCG`, `NewtonCG`, `TrustNCG`, `TrustKrylov`, `Dogleg`
- Bounded physical-space defaults: `LBFGSB`, `TrustConstr`, `TNC`, `SLSQP`, `COBYQA`, `COBYLA`, `SR1`

See `src/dfbench/algorithms/gradient_based/scipy/_common.py` and `scripts/voyager_scipy_benchmark.py` for the shared wrapper and a benchmark example.

---

### Native-JAX Custom/Hybrid Batch

These classes are implemented as lightweight, benchmark-oriented methods that
stay fully in JAX and use Objective logging directly:

- `SGLDJAX`: optimizer-style SGLD (not full Bayesian posterior sampling)
- `ASAMJAX`: adaptive-SAM style adversarial smoothing
- `AdamToLBFGSJAX`: Adam exploration then Optax L-BFGS refinement
- `EntropySGDJAX`: minimal local-entropy inner loop
- `SGHMCJAX`: momentum + friction + noise stochastic dynamics
- `OGDJAX`, `OAdamJAX`: optimistic gradient and optimistic Adam variants
- `PerturbedGDJAX`, `NoisyAdamJAX`: simple ruggedness controls
- `GDRestartsJAX`: GD with first-class periodic restarts
- `GaussianSmoothingGDJAX`: antithetic Gaussian smoothing + GD

All methods above default to unbounded optimization coordinates (`unbounded=True`).
Restart controls are exposed as conservative hyperparameters where applicable.

`ARCJAX` is currently intentionally disabled and raises `NotImplementedError`
to fail loudly until a stable and benchmark-fair implementation is ready.

---

## Evolutionary Algorithms

These algorithms search directly in the bounded parameter space using population-based strategies. They use `obj.vmap_value()` for efficient batch evaluation.

### RandomSearch

Simplest baseline. Draws uniform random samples within bounds and evaluates them in batches.

```python
optimizer = RandomSearch(batch_size=100)
optimizer.optimize(
    problem_objective=obj,
    max_iterations=None,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `batch_size` | `100` | Samples per batch. |

**Rationale — why include random search?** It serves as the baseline for all other algorithms. If a sophisticated method can't beat random search, something is wrong with its configuration or the problem is too easy to differentiate methods.

---

### EvoxPSO (Particle Swarm Optimization)

Uses the [EvoX](https://github.com/EMI-Group/evox) library's PSO implementations with PyTorch backend. Supports multiple PSO variants.

```python
optimizer = EvoxPSO(batch_size=5, variant="CLPSO")
optimizer.optimize(
    problem_objective=obj,
    pop_size=200,
    n_generations=10000,
    random_seed=42,
)
```

| Init parameter | Default | Description |
|----------------|---------|-------------|
| `batch_size` | `5` | Particles evaluated simultaneously per sub-batch. Reduce if running out of GPU memory. |
| `variant` | `"PSO"` | Algorithm variant (see below). |

| `optimize()` parameter | Default | Description |
|------------------------|---------|-------------|
| `pop_size` | `100` | Number of particles in the swarm. |
| `n_generations` | `10000` | Maximum generations. |

**Available variants:**

| Variant | Full name |
|---------|-----------|
| `PSO` | Standard Particle Swarm Optimization |
| `CLPSO` | Comprehensive Learning PSO |
| `CSO` | Competitive Swarm Optimizer |
| `DMSPSOEL` | Dynamic Multi-Swarm PSO with Elite Learning |
| `FSPSO` | Fitness-Sharing PSO |
| `SLPSOGS` | Social Learning PSO with Gaussian Sampling |
| `SLPSOUS` | Social Learning PSO with Uniform Sampling |

**Implementation detail:** Because EvoX uses PyTorch tensors and the objective is JAX-based, the algorithm internally converts between frameworks using `t2j` / `j2t`. Particles are evaluated in mini-batches of size `batch_size` to control GPU memory usage.

---

### EvoxES (Evolution Strategies — EvoX backend)

Uses EvoX's evolution strategy implementations. Similar structure to EvoxPSO but with different algorithmic families.  Note: the EvoX backend is distinct from the pycma / cmaes / evosax backends added in the CMA-family batch.

```python
optimizer = EvoxES(batch_size=5, variant="CMAES")
optimizer.optimize(
    problem_objective=obj,
    pop_size=100,
    n_generations=10000,
    random_seed=42,
)
```

**Available variants:**

| Variant | Full name |
|---------|-----------|
| `CMAES` | Covariance Matrix Adaptation Evolution Strategy |
| `OpenES` | OpenAI Evolution Strategy |
| `XNES` | Exponential Natural Evolution Strategy |
| `SeparableNES` | Separable Natural Evolution Strategy |
| `DES` | Distributed Evolution Strategy |
| `SNES` | Separable NES |
| `ARS` | Augmented Random Search |
| `ASEBO` | Adaptive Sampling Evolution-Based Optimization |
| `PersistentES` | Persistent Evolution Strategy |
| `NoiseReuseES` | Noise Reuse Evolution Strategy |
| `GuidedES` | Guided Evolution Strategy |
| `ESMC` | Evolution Strategy with Monte Carlo |

---

## Direct Search Algorithms

Derivative-free mesh-based algorithms that operate in **bounded physical space** and refine a mesh/poll structure around the incumbent point. These are local explorers for rugged landscapes — not global optimizers. Uses the [OMADS](https://github.com/Ahmed-Bayoumy/OMADS) library.

### OmadsMADS (Mesh Adaptive Direct Search)

Full MADS algorithm with search step (broad sampling) and poll step (structured directions). Each iteration first samples the mesh, then polls orthogonal directions. The mesh refines on failure and coarsens on success.

```python
optimizer = OmadsMADS(psize_init=1.0, tol=1e-9, ns=4)
optimizer.optimize(
    problem_objective=obj,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `psize_init` | `1.0` | Initial poll-step (frame) size. |
| `tol` | `1e-9` | Convergence tolerance on mesh/frame size. |
| `ns` | `4` | Number of search samples per search step. |

---

### OmadsOrthoMADS (Orthogonal MADS, poll only)

Runs only the OrthoMADS poll step with orthogonal Householder directions. Leaner per-iteration cost than full MADS, tighter local convergence.

```python
optimizer = OmadsOrthoMADS(psize_init=1.0, tol=1e-9)
optimizer.optimize(
    problem_objective=obj,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `psize_init` | `1.0` | Initial poll-step (frame) size. |
| `tol` | `1e-9` | Convergence tolerance on mesh/frame size. |

---

### Nevergrad Baselines (Rugged-Landscape Controls)

A small batch of [Nevergrad](https://github.com/facebookresearch/nevergrad) wrappers intended as rugged-landscape controls. All operate in **bounded physical space** and evaluate candidates through the `Objective` for fair benchmark accounting.

#### NevergradOnePlusOne

Lightweight (1+1)-ES: maintains a single candidate, perturbs it with Gaussian noise, and accepts only improvements. Minimal overhead, useful as a sanity-check baseline.

```python
optimizer = NevergradOnePlusOne()
optimizer.optimize(
    problem_objective=obj,
    n_restarts=3,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `n_restarts` | `1` | Independent restarts (budget split evenly). |
| `max_iterations` | `None` | Total ask/tell cap across restarts. |

---

#### NevergradTBPSA

Test-Based Population-Size Adaptation. A noise-robust baseline that dynamically adapts its population size. Supports repeated evaluations per candidate for noise averaging.

```python
optimizer = NevergradTBPSA()
optimizer.optimize(
    problem_objective=obj,
    n_restarts=1,
    num_evaluations=3,    # average 3 evaluations per candidate
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `n_restarts` | `1` | Independent restarts. |
| `num_evaluations` | `1` | Repeated evals per candidate (averaged). Each counts against budget. |
| `max_iterations` | `None` | Total ask/tell cap across restarts. |

**Rationale — repeated evaluations:** On noisy landscapes, averaging multiple evaluations per candidate gives the optimizer a more stable signal. Set `num_evaluations > 1` when evaluation noise is suspected.

---

#### NevergradNGOpt

Nevergrad's automatic algorithm-selection meta-optimizer. Internally chooses and configures an algorithm based on problem characteristics (budget, dimensionality). Serves as a strong library-default baseline without manual tuning.

```python
optimizer = NevergradNGOpt()
optimizer.optimize(
    problem_objective=obj,
    n_restarts=1,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `n_restarts` | `1` | Independent restarts. |
| `max_iterations` | `None` | Total ask/tell cap across restarts. |

**Rationale — why include NGOpt?** It represents Nevergrad's best automatic guess for a given problem. Comparing it against hand-tuned algorithms reveals whether manual algorithm selection adds value.

---

## Powell-Style Trust-Region DFO (PDFO / Py-BOBYQA)

Model-based derivative-free trust-region solvers by M. J. D. Powell.  All run in **bounded physical space** with multistart restarts.  Each call solves a quadratic model in a shrinking trust region; convergence is local but very precise on smooth landscapes.

**Required packages** (install with `uv add pdfo Py-BOBYQA`):
- `pdfo` — for `PDFOUOBYQA`, `PDFONEWUOA`, `PDFOLINCOA`
- `Py-BOBYQA` — for `PyBOBYQA`

```python
from dfbench.algorithms import PDFOUOBYQA, PDFONEWUOA, PDFOLINCOA, PyBOBYQA
```

| Algorithm | Constraint support | Notes |
|-----------|-------------------|-------|
| `PDFOUOBYQA` | unconstrained | Quadratic interpolation, ``2n+1`` points. |
| `PDFONEWUOA` | unconstrained | Powell's NEWUOA, sparser interpolation. |
| `PDFOLINCOA` | bounds + linear | Reads `problem.linear_constraints` (`A_ub @ x <= b_ub`) when present. |
| `PyBOBYQA`  | bounds | BOBYQA with optional softmax-style restart heuristics. |

Common hyperparameters (passed at construction):

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `radius_init` | 10% of mean bound range | Initial trust-region radius. |
| `radius_final` | `1e-6` | Convergence tolerance on radius. |
| `npt` | `2*n+1` | Number of interpolation points (PDFO solvers). |
| `n_restarts` | `1` | Multistart restarts within evaluation budget. |

```python
optimizer = PDFOLINCOA(radius_init=0.5, n_restarts=3)
optimizer.optimize(problem_objective=obj, random_seed=42)
```

---

## CMA-Family Algorithms (pycma / cmaes / evosax / native JAX)

Nine additional CMA-family algorithms added alongside the EvoX backend.  Each class names its backend explicitly in `algorithm_str` so benchmark runs can be distinguished.

**Required packages** (install with `uv add cma cmaes evosax`):
- `pycma` ≥ 3.3 — for `PyCMA*` classes
- `cmaes` ≥ 0.10 — for `CMAESSepCMA`
- `evosax` ≥ 0.1.6 — for `Evosax*` classes
- `jax` (already a dependency) — for `JAX*` classes

```python
from dfbench.algorithms import (
    PyCMACMAES, PyCMAActiveCMAES, PyCMAIPOP, PyCMABIPOP,  # pycma
    CMAESSepCMA,                                            # cmaes
    EvosaxMAES, EvosaxLMMAES,                               # evosax
    JAXOnePlusOneES, JAXMuLambdaES,                         # native JAX
)
```

### PyCMACMAES (pycma — vanilla CMA-ES)

```python
optimizer = PyCMACMAES(batch_size=50)
optimizer.optimize(obj, pop_size=50, sigma0=0.5, max_iterations=500, random_seed=0)
```

| parameter | default | description |
|-----------|---------|-------------|
| `batch_size` | `1` | Candidates per `vmap_value` call (constructor). |
| `pop_size` | `4+floor(3·ln n)` | Population size λ (optimize). |
| `sigma0` | `0.3·mean(ub−lb)` | Initial step size (optimize). |
| `max_iterations` | `None` | Generation cap (optimize). |

### PyCMAActiveCMAES (pycma — aCMA-ES)

Identical to `PyCMACMAES` with `CMA_active=True`.  Uses negative weight updates for unsuccessful directions.

```python
optimizer = PyCMAActiveCMAES(batch_size=50)
optimizer.optimize(obj, pop_size=50, random_seed=0)
```

### PyCMAIPOP (pycma — IPOP-CMA-ES)

Restarts CMA-ES up to `max_restarts` times, doubling λ each time.

```python
optimizer = PyCMAIPOP(batch_size=20)
optimizer.optimize(obj, pop_size=20, max_restarts=5, random_seed=0, max_iterations_per_restart=200)
```

| parameter | default | description |
|-----------|---------|-------------|
| `batch_size` | `1` | Candidates per `vmap_value` call (constructor). |
| `pop_size` | `4+floor(3·ln n)` | Base λ (doubles each restart) (optimize). |
| `max_restarts` | `9` | Maximum restarts (optimize). |
| `max_iterations_per_restart` | `None` | Per-restart generation cap (optimize). |

### PyCMABIPOP (pycma — BIPOP-CMA-ES)

Alternates between large-population (doubled λ) and small-population (random λ, random σ) restarts following Hansen 2009.

```python
optimizer = PyCMABIPOP(batch_size=20)
optimizer.optimize(obj, pop_size=20, max_restarts=10, random_seed=0)
```

### CMAESSepCMA (cmaes package — sep-CMA-ES)

Diagonal covariance matrix; O(n²) instead of O(n³) per update.

```python
optimizer = CMAESSepCMA(batch_size=50)
optimizer.optimize(obj, pop_size=50, sigma0=0.5, max_no_improvement=100, random_seed=0)
```

| parameter | default | description |
|-----------|---------|-------------|
| `batch_size` | `1` | Candidates per `vmap_value` call (constructor). |
| `pop_size` | library default | Population λ (optimize). |
| `max_no_improvement` | `None` | Stop on stagnation after N generations (optimize). |

### EvosaxMAES (evosax — MA-ES)

Matrix Adaptation ES via the evosax JAX library.

```python
optimizer = EvosaxMAES(batch_size=64)
optimizer.optimize(obj, pop_size=64, sigma0=0.3, max_iterations=1000, random_seed=0)
```

### EvosaxLMMAES (evosax — LM-MA-ES)

Limited-memory MA-ES; O(n·m) storage where m is `memory_size`.

```python
optimizer = EvosaxLMMAES(batch_size=64)
optimizer.optimize(obj, pop_size=64, memory_size=10, random_seed=0)
```

### JAXOnePlusOneES (native JAX — (1+1)-ES)

Single-parent ES with the 1/5 success rule.  No optional dependencies.

```python
optimizer = JAXOnePlusOneES()
optimizer.optimize(obj, sigma0=0.3, sigma_min=1e-10, success_window=20, max_iterations=5000, random_seed=0)
```

### JAXMuLambdaES (native JAX — (μ,λ)-ES)

Comma-selection ES with isotropic Gaussian mutations and cumulative step-size adaptation.  No optional dependencies.

```python
optimizer = JAXMuLambdaES(batch_size=50)
optimizer.optimize(obj, mu=10, lam=50, sigma0=0.3, sigma_min=1e-10, max_iterations=500, random_seed=0)
```

| parameter | default | description |
|-----------|---------|-------------|
| `batch_size` | `1` | Candidates per `vmap_value` call (constructor). |
| `mu` | `10` | Number of survivors (must be < lam) (optimize). |
| `lam` | `50` | Number of offspring per generation (optimize). |

---

## Surrogate-Based Algorithms

These algorithms build a surrogate model of the loss landscape and use it to select promising evaluation points.

### BotorchBO (Bayesian Optimization)

Standard Bayesian Optimization using a Gaussian Process surrogate and batch Expected Improvement acquisition (qLogEI).

```python
optimizer = BotorchBO()
optimizer.optimize(
    problem_objective=obj,
    max_iterations=100,      # required
    n_initial=10,            # Sobol samples before fitting GP
    batch_size=1,            # points per acquisition
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `max_iterations` | *required* | BO iterations (excluding initial samples). |
| `n_initial` | `10` | Initial Sobol quasi-random samples. |
| `batch_size` | `1` | Points acquired per iteration. |

**Implementation detail:** Parameters are internally normalized to $[0, 1]$. The GP is fit on negated losses (BoTorch maximizes by convention). NaN/Inf evaluations are retried with small perturbations up to `max_retries` times.

---

### BotorchTuRBO (Trust Region BO)

Implements TuRBO-1 from [Eriksson et al. 2019](https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7e0d4b91ac0c4ed2f54-Abstract.html). Maintains a local trust region that expands on success and shrinks on failure, making it effective for high-dimensional problems where global BO struggles.

```python
optimizer = BotorchTuRBO()
optimizer.optimize(
    problem_objective=obj,
    max_iterations=100,
    n_initial=20,
    batch_size=5,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `max_iterations` | *required* | TuRBO iterations. |
| `n_initial` | `20` | Initial Sobol samples. |
| `batch_size` | `5` | Points per acquisition. |

**Trust region mechanics:**
- After `success_tolerance` consecutive improvements → region **doubles** in size.
- After `failure_tolerance` consecutive non-improvements → region **halves** in size.
- When region shrinks below `length_min` → **restart** from scratch (re-initialize Sobol samples).

---

### ReSTIR (Resampled Surrogate-based Importance Sampling)

A kNN-surrogate-based algorithm implemented in pure JAX. Uses k-nearest-neighbors regression to estimate the loss surface and importance sampling to focus evaluations on promising regions.

```python
from dfbench.algorithms.surrogate_based.restir import MyAlgorithm as ReSTIR

optimizer = ReSTIR(batch_size=100)
optimizer.optimize(
    problem_objective=obj,
    n_initial_samples=1000,
    n_knn_samples=100_000,
    k_neighbors=10,
    temperature=1.0,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `n_initial_samples` | `1000` | Initial random samples for training the kNN. |
| `n_knn_samples` | `100_000` | Candidate samples generated per iteration. |
| `k_neighbors` | `10` | Neighbors for kNN regression. |
| `temperature` | `1.0` | Controls exploration vs exploitation. Higher = more exploration. |

**Rationale — why kNN instead of GP?** GPs have $O(n^3)$ fitting cost, making them impractical for large training sets. kNN regression with inverse-distance weighting runs in $O(n)$ per query via JAX's `top_k`, scales to 100k+ candidates, and stays entirely on GPU.

---

## Generative Algorithms

### VAESampling

Two-phase approach: (1) train a Variational Autoencoder on high-quality samples to learn a compressed latent space, then (2) run Bayesian Optimization in that latent space.

```python
optimizer = VAESampling()
optimizer.optimize(
    problem_objective=obj,
    max_iterations=50,
    vae_training_samples=1000,
    vae_epochs=100,
    batch_size=64,
    random_seed=42,
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `vae_training_samples` | `1000` | Samples for VAE training. |
| `vae_epochs` | `100` | Training epochs with cyclic KL annealing. |
| `batch_size` | `64` | Mini-batch size for VAE training. |
| `max_iterations` | *required* | BO iterations in latent space. |

**Architecture details:**
- ResNet-style VAE with residual blocks, batch normalization, and Mish activations.
- Latent dimension = `n_params / 10` (compressed 10×).
- Cyclic $\beta$-annealing for stable training.
- After training, BO uses `qLogEI` acquisition in the learned latent space.

**Rationale — why compress to latent space?** High-dimensional BO suffers from the curse of dimensionality. The VAE learns which parameter combinations matter, projecting the search into a much lower-dimensional space where the GP surrogate is more effective.

---

## Optax Optimizer Batch (34 algorithms)

All Optax-based algorithms share a common base class `OptaxAlgorithm` and live in `src/dfbench/algorithms/gradient_based/optax/`. They operate in **unbounded** (sigmoid-transformed) space and use `obj.value_and_grad()` for gradient information.

**Import:**

```python
from dfbench.algorithms import (
    OptaxAdam, OptaxAdamW, OptaxAdaBelief, OptaxAdafactor,
    OptaxAMSGrad, OptaxAdaGrad, OptaxAdaDelta, OptaxAdaMax,
    OptaxAdaMaxW, OptaxAdan, OptaxLion, OptaxLAMB,
    OptaxNadam, OptaxNadamW, OptaxRMSProp, OptaxRProp,
    OptaxRAdam, OptaxSGD, OptaxSGDM, OptaxNAG,
    OptaxNoisySGD, OptaxPolyakSGD, OptaxSAM, OptaxSophia,
    OptaxLookahead, OptaxScheduleFreeAdam, OptaxYogi,
    OptaxNovoGrad, OptaxOGD, OptaxOAdam,
    OptaxSignSGD, OptaxSignum, OptaxSM3, OptaxLBFGS,
)
```

**Shared hyperparameters.** All standard-loop algorithms accept:

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `learning_rate` | `0.1` | Base learning rate. |
| `grad_clip_norm` | `1.0` | Maximum global gradient L2 norm (`None` to disable). |
| `patience` | `None` | Early-stop after this many evals without improvement. |

Additional algorithm-specific hyperparameters are passed as keyword arguments to `optimize()`. The shared helper `build_optimizer()` provides optional gradient clipping and learning-rate warmup.

---

### OptaxAdam

Standard Adam optimizer ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)).

```python
optimizer = OptaxAdam()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAdamW

Adam with decoupled weight decay ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)).

```python
optimizer = OptaxAdamW()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, weight_decay=1e-4, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `weight_decay` | `1e-4` | Decoupled weight decay coefficient. |

---

### OptaxAdaBelief

AdaBelief — adapts step sizes based on *belief* in the gradient ([Zhuang et al., 2020](https://arxiv.org/abs/2010.07468)).

```python
optimizer = OptaxAdaBelief()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAdafactor

Memory-efficient factored Adam ([Shazeer & Stern, 2018](https://arxiv.org/abs/1804.04235)).

```python
optimizer = OptaxAdafactor()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAMSGrad

AMSGrad variant of Adam that maintains the maximum of past squared gradients ([Reddi et al., 2018](https://arxiv.org/abs/1904.09237)).

```python
optimizer = OptaxAMSGrad()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAdaGrad

Adaptive gradient method — per-parameter learning rates decay based on accumulated squared gradients ([Duchi et al., 2011](https://jmlr.org/papers/v12/duchi11a.html)).

```python
optimizer = OptaxAdaGrad()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAdaDelta

AdaDelta — learning-rate-free adaptive method using running averages ([Zeiler, 2012](https://arxiv.org/abs/1212.5701)).

```python
optimizer = OptaxAdaDelta()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAdaMax

AdaMax — $L^\infty$ variant of Adam ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)).

```python
optimizer = OptaxAdaMax()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxAdaMaxW

AdaMax with decoupled weight decay.

```python
optimizer = OptaxAdaMaxW()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, weight_decay=1e-4, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `weight_decay` | `1e-4` | Decoupled weight decay coefficient. |

---

### OptaxAdan

Adaptive Nesterov momentum algorithm ([Xie et al., 2023](https://arxiv.org/abs/2208.06677)).

```python
optimizer = OptaxAdan()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxLion

Evolved sign momentum optimizer — discovered via meta-learning ([Chen et al., 2023](https://arxiv.org/abs/2302.06675)).

```python
optimizer = OptaxLion()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxLAMB

Layer-wise Adaptive Moments for Batch training ([You et al., 2020](https://arxiv.org/abs/1904.00962)).

```python
optimizer = OptaxLAMB()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxNadam

Nesterov-accelerated Adam ([Dozat, 2016](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)).

```python
optimizer = OptaxNadam()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxNadamW

Nadam with decoupled weight decay.

```python
optimizer = OptaxNadamW()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, weight_decay=1e-4, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `weight_decay` | `1e-4` | Decoupled weight decay coefficient. |

---

### OptaxRMSProp

RMSProp — root mean square propagation ([Hinton, 2012](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)).

```python
optimizer = OptaxRMSProp()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxRProp

RProp — resilient backpropagation with sign-based updates ([Riedmiller & Braun, 1993](https://ieeexplore.ieee.org/document/298623)).

```python
optimizer = OptaxRProp()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxRAdam

Rectified Adam — variance-rectified adaptive learning rate ([Liu et al., 2020](https://arxiv.org/abs/1908.03265)).

```python
optimizer = OptaxRAdam()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxSGD / OptaxSGDM / OptaxNAG

Vanilla SGD, SGD with Momentum, and Nesterov Accelerated Gradient. All three live in the same file (`optax_sgd.py`) and share the standard loop.

```python
OptaxSGD().optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
OptaxSGDM().optimize(problem_objective=obj, learning_rate=0.1, momentum=0.9, random_seed=42)
OptaxNAG().optimize(problem_objective=obj, learning_rate=0.1, momentum=0.9, random_seed=42)
```

| Extra hyperparameter | Default | Applies to | Description |
|----------------------|---------|------------|-------------|
| `momentum` | `0.9` | SGDM, NAG | Momentum coefficient. |

---

### OptaxNoisySGD

SGD with decaying Gaussian noise injection.

```python
optimizer = OptaxNoisySGD()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, eta=0.01, gamma=0.55, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `eta` | `0.01` | Noise scale. |
| `gamma` | `0.55` | Noise decay exponent. |

---

### OptaxPolyakSGD

Polyak step-size SGD — adapts step size using $\text{step} = (f(x) - f^*) / \lVert g \rVert^2$.

> **Note:** Requires passing the current loss to `optimizer.update()`, so this algorithm uses a custom loop.

```python
optimizer = OptaxPolyakSGD()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, f_min=0.0, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `max_learning_rate` | `learning_rate` | Maximum step size. |
| `f_min` | `0.0` | Estimated optimal value $f^*$. |

---

### OptaxSAM

Sharpness-Aware Minimization — seeks flat minima by perturbing towards the worst-case neighbourhood ([Foret et al., 2021](https://arxiv.org/abs/2010.01412)).

> **Note:** Each SAM iteration uses **two** `value_and_grad` evaluations (one adversarial, one descent). This algorithm overrides the standard loop.

```python
optimizer = OptaxSAM()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, rho=0.05, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `rho` | `0.05` | Adversarial perturbation radius. |
| `sync_period` | `2` | Steps between adversarial and descent phases. |

---

### OptaxSophia

Sophia optimizer — lightweight second-order method using diagonal Hessian EMA with element-wise clipping ([Liu et al., 2023](https://arxiv.org/abs/2305.14342)).

> Optax 0.2.4 does not include Sophia natively. A local `GradientTransformation` wrapper implements Sophia-G (squared-gradient Hessian approximation).

```python
optimizer = OptaxSophia()
optimizer.optimize(problem_objective=obj, learning_rate=1e-3, gamma=0.01, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `b1` | `0.965` | First moment decay. |
| `b2` | `0.99` | Hessian diagonal EMA decay. |
| `gamma` | `0.01` | Clipping threshold — updates clipped to $[-1/\gamma, 1/\gamma]$. |
| `weight_decay` | `0.0` | Decoupled weight decay. |

---

### OptaxLookahead

Lookahead wrapper — slow-weight averaging around a fast inner optimizer ([Zhang et al., 2019](https://arxiv.org/abs/1907.08610)).

> Uses `optax.LookaheadParams` internally to maintain fast and slow weights.

```python
optimizer = OptaxLookahead()
optimizer.optimize(
    problem_objective=obj,
    learning_rate=0.1,
    inner_optimizer_name="adam",  # adam | adamw | sgd | rmsprop | lion
    sync_period=6,
    slow_step_size=0.5,
    random_seed=42,
)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `inner_optimizer_name` | `"adam"` | Inner optimizer: `adam`, `adamw`, `sgd`, `rmsprop`, `lion`. |
| `sync_period` | `6` | Fast-weight steps between slow-weight syncs (k). |
| `slow_step_size` | `0.5` | Interpolation factor $\alpha$ for slow update. |

---

### OptaxScheduleFreeAdam

Schedule-Free Adam — removes the need for an explicit LR schedule by maintaining two parameter sequences ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)).

```python
optimizer = OptaxScheduleFreeAdam()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, warmup_steps=200, random_seed=42)
```

| Extra hyperparameter | Default | Description |
|----------------------|---------|-------------|
| `warmup_steps` | `200` | Linear warmup length. |

---

### OptaxYogi

Yogi optimizer — controls adaptive learning-rate increase more conservatively than Adam ([Zaheer et al., 2018](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization)).

```python
optimizer = OptaxYogi()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxNovoGrad

NovoGrad — layer-wise gradient normalization optimizer ([Ginsburg et al., 2019](https://arxiv.org/abs/1905.11286)).

```python
optimizer = OptaxNovoGrad()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxOGD / OptaxOAdam

Optimistic GD and Optimistic Adam.

```python
OptaxOGD().optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
OptaxOAdam().optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxSignSGD / OptaxSignum

Sign-based optimizers — update with the sign of the gradient.

```python
OptaxSignSGD().optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
OptaxSignum().optimize(problem_objective=obj, learning_rate=0.1, momentum=0.9, random_seed=42)
```

| Extra hyperparameter | Default | Applies to | Description |
|----------------------|---------|------------|-------------|
| `momentum` | `0.9` | Signum | Momentum coefficient. |

---

### OptaxSM3

SM3 — memory-efficient adaptive optimizer for sparse gradients ([Anil et al., 2019](https://arxiv.org/abs/1901.11150)).

```python
optimizer = OptaxSM3()
optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)
```

---

### OptaxLBFGS

L-BFGS optimizer via Optax. Uses second-order curvature information for faster convergence on smooth landscapes.

> Same JIT-compiled pattern as `LBFGSGD` but registered under the `optax_*` naming scheme. Passes the raw value function and gradients to `optimizer.update()` for internal line-search.

```python
optimizer = OptaxLBFGS()
optimizer.optimize(problem_objective=obj, patience=500, random_seed=42)
```

---

### Optax Batch Summary

| Class | `algorithm_str` | Key property | Custom loop? |
|-------|-----------------|--------------|:------------:|
| `OptaxAdam` | `optax_adam` | Standard Adam | — |
| `OptaxAdamW` | `optax_adamw` | Decoupled weight decay | — |
| `OptaxAdaBelief` | `optax_adabelief` | Belief-based adaptation | — |
| `OptaxAdafactor` | `optax_adafactor` | Memory-efficient factored | — |
| `OptaxAMSGrad` | `optax_amsgrad` | Max past squared gradients | — |
| `OptaxAdaGrad` | `optax_adagrad` | Per-param accumulated LR | — |
| `OptaxAdaDelta` | `optax_adadelta` | LR-free adaptive | — |
| `OptaxAdaMax` | `optax_adamax` | $L^\infty$ Adam | — |
| `OptaxAdaMaxW` | `optax_adamaxw` | AdaMax + weight decay | — |
| `OptaxAdan` | `optax_adan` | Nesterov momentum variant | — |
| `OptaxLion` | `optax_lion` | Evolved sign momentum | — |
| `OptaxLAMB` | `optax_lamb` | Layer-wise adaptive | — |
| `OptaxNadam` | `optax_nadam` | Nesterov Adam | — |
| `OptaxNadamW` | `optax_nadamw` | Nadam + weight decay | — |
| `OptaxRMSProp` | `optax_rmsprop` | Root mean square prop | — |
| `OptaxRProp` | `optax_rprop` | Resilient backprop | — |
| `OptaxRAdam` | `optax_radam` | Rectified Adam | — |
| `OptaxSGD` | `optax_sgd` | Vanilla SGD | — |
| `OptaxSGDM` | `optax_sgdm` | SGD + momentum | — |
| `OptaxNAG` | `optax_nag` | Nesterov accelerated | — |
| `OptaxNoisySGD` | `optax_noisy_sgd` | Gaussian noise injection | — |
| `OptaxPolyakSGD` | `optax_polyak_sgd` | Polyak step-size | yes |
| `OptaxSAM` | `optax_sam` | Sharpness-aware (2× grad) | yes |
| `OptaxSophia` | `optax_sophia` | Diagonal Hessian clipping | — |
| `OptaxLookahead` | `optax_lookahead` | Slow-weight averaging | yes |
| `OptaxScheduleFreeAdam` | `optax_schedule_free_adam` | Schedule-free | — |
| `OptaxYogi` | `optax_yogi` | Conservative adaptive LR | — |
| `OptaxNovoGrad` | `optax_novograd` | Layer-wise grad norm | — |
| `OptaxOGD` | `optax_ogd` | Optimistic GD | — |
| `OptaxOAdam` | `optax_oadam` | Optimistic Adam | — |
| `OptaxSignSGD` | `optax_sign_sgd` | Sign of gradient | — |
| `OptaxSignum` | `optax_signum` | Sign + momentum | — |
| `OptaxSM3` | `optax_sm3` | Memory-efficient sparse | — |
| `OptaxLBFGS` | `optax_lbfgs` | Quasi-Newton (JIT loop) | yes |

---

## Summary Table

| Algorithm | Type | Key strength | Typical use case |
|-----------|------|-------------|------------------|
| `AdamGD` | Gradient | Fast convergence on smooth landscapes | Quick prototyping, smooth problems |
| `SAGD` | Gradient | Escapes local minima via stochastic ascent | Rugged landscapes |
| `NAAdamGD` | Gradient | Noise-based exploration with annealing | Balancing exploration and exploitation |
| `LBFGSGD` | Gradient | Second-order curvature | Smooth, well-conditioned problems |
| `OptaxAdam` — `OptaxLBFGS` | Gradient | 34 Optax optimizers (see table above) | Systematic algorithm comparison |
| `RandomSearch` | Evolutionary | No hyperparameters, unbiased baseline | Baseline comparison |
| `EvoxPSO` | Evolutionary | Swarm intelligence, many variants | Moderate-dimensional problems |
| `EvoxES` | Evolutionary | Covariance adaptation (CMA-ES) | General black-box optimization |
| `NevergradOnePlusOne` | Evolutionary | Minimal (1+1)-ES, very lightweight | Sanity-check baseline |
| `NevergradTBPSA` | Evolutionary | Noise-robust, adaptive population | Noisy / rugged landscapes |
| `NevergradNGOpt` | Evolutionary | Auto algorithm selection | Library-default baseline |
| `BotorchBO` | Surrogate | Sample-efficient, uncertainty-aware | Low evaluation budgets |
| `BotorchTuRBO` | Surrogate | Local trust region, high-dim friendly | High-dimensional, expensive evals |
| `ReSTIR` | Surrogate | Scalable kNN surrogate, GPU-native | Large candidate pools |
| `OmadsMADS` | Direct Search | MADS search + poll, mesh refinement | Rugged-landscape local exploration |
| `OmadsOrthoMADS` | Direct Search | OrthoMADS poll only, orthogonal dirs | Local refinement, predictable cost |
| `VAESampling` | Generative | Latent-space compression | Very high-dimensional problems |
