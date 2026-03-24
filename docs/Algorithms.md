# Algorithms

All built-in algorithms subclass `OptimizationAlgorithm` and follow a common contract: they receive a pre-configured `Objective`, run their optimization loop, and mutate it in place (logging all results). `optimize()` returns `None` — the caller accesses results from the same `Objective` instance it passed in.

**Import:**

```python
from dfbench.algorithms import (
    AdamGD, SAGD, NAAdamGD, LBFGSGD,       # gradient-based
    RandomSearch, EvoxPSO, EvoxES,           # evolutionary
    NevergradOnePlusOne, NevergradTBPSA,     # nevergrad baselines
    NevergradNGOpt,                          # nevergrad meta-optimizer
    BotorchBO, BotorchTuRBO,                 # surrogate-based
    VAESampling,                              # generative
)
```

---

## Algorithm Types

Every algorithm declares an `algorithm_type` from the `AlgorithmType` enum:

| Type | `unbounded` | Evaluation methods used | Examples |
|------|-------------|------------------------|----------|
| `GRADIENT_BASED` | `True` | `value_and_grad()` | Adam, SA-GD, NA-Adam, L-BFGS |
| `EVOLUTIONARY` | `False` | `vmap_value()` | Random Search, PSO, CMA-ES |
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
| `noise_anneal_iters` | `5000` | Iterations over which noise decays. |
| `noise_cap_relative_to_update` | `0.25` | Caps noise to this fraction of the Adam update norm. |
| `noise_cap_start_iter` | `500` | Iteration at which relative capping activates. |

**Rationale — noise capping:** Without capping, noise can overwhelm the optimizer update when gradients are very small (near a plateau). The relative cap ensures noise never exceeds 25% (by default) of the Adam step magnitude.

---

### LBFGSGD

L-BFGS optimizer from Optax. Uses second-order curvature information for faster convergence on smooth landscapes.

> **Note:** Because `optax.lbfgs` needs the raw value function for its internal line-search, this algorithm JIT-compiles the full optimization step and uses `obj.log_evaluation()` to record results after each step (instead of calling `obj.value_and_grad()` directly). This makes it a useful reference for implementing other algorithms that require custom JIT-compiled evaluation loops — see `src/dfbench/algorithms/gradient_based/lbfgs_gd.py`.

```python
optimizer = LBFGSGD()
optimizer.optimize(
    problem_objective=obj,
    patience=500,
    random_seed=42,
)
```

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

### EvoxES (Evolution Strategies)

Uses EvoX's evolution strategy implementations. Similar structure to EvoxPSO but with different algorithmic families.

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

## Summary Table

| Algorithm | Type | Key strength | Typical use case |
|-----------|------|-------------|------------------|
| `AdamGD` | Gradient | Fast convergence on smooth landscapes | Quick prototyping, smooth problems |
| `SAGD` | Gradient | Escapes local minima via stochastic ascent | Rugged landscapes |
| `NAAdamGD` | Gradient | Noise-based exploration with annealing | Balancing exploration and exploitation |
| `LBFGSGD` | Gradient | Second-order curvature | Smooth, well-conditioned problems |
| `RandomSearch` | Evolutionary | No hyperparameters, unbiased baseline | Baseline comparison |
| `EvoxPSO` | Evolutionary | Swarm intelligence, many variants | Moderate-dimensional problems |
| `EvoxES` | Evolutionary | Covariance adaptation (CMA-ES) | General black-box optimization |
| `NevergradOnePlusOne` | Evolutionary | Minimal (1+1)-ES, very lightweight | Sanity-check baseline |
| `NevergradTBPSA` | Evolutionary | Noise-robust, adaptive population | Noisy / rugged landscapes |
| `NevergradNGOpt` | Evolutionary | Auto algorithm selection | Library-default baseline |
| `BotorchBO` | Surrogate | Sample-efficient, uncertainty-aware | Low evaluation budgets |
| `BotorchTuRBO` | Surrogate | Local trust region, high-dim friendly | High-dimensional, expensive evals |
| `ReSTIR` | Surrogate | Scalable kNN surrogate, GPU-native | Large candidate pools |
| `VAESampling` | Generative | Latent-space compression | Very high-dimensional problems |
