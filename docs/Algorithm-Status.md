# Algorithm Implementation Status

> Overview of all implemented algorithms across branches, sorted by completion status and grouped by category.
>
> **Finished** = merged into `main` and tested. All other tiers indicate work-in-progress on feature branches.
>
> All branches listed below have been verified to merge cleanly with `main` (0 conflicts via `git merge-tree`).

---

## Status Legend

| Status | Meaning |
|--------|---------|
| **Finished** | On `main`, tested, production-ready |
| **Branch** | On a feature branch, merges cleanly with `main` (0 conflicts verified) |
| **Future branch** | On the `future` integration branch |

---

## `EvoxES` — Multi-Algorithm Wrapper (on `main`)

`EvoxES` is a single class that wraps **12 distinct EvoX evolution strategies** selectable via its `variant` parameter.
Each variant is a separate algorithm from the EvoX library.

| Variant | Full Name | Type |
|---------|-----------|------|
| `CMAES` | Covariance Matrix Adaptation ES | ES |
| `OpenES` | OpenAI Evolution Strategy | ES |
| `XNES` | Exponential Natural Evolution Strategy | NES |
| `SeparableNES` | Separable Natural Evolution Strategy | NES |
| `DES` | Distributed Evolution Strategy | ES |
| `SNES` | Separable NES | NES |
| `ARS` | Augmented Random Search | ES |
| `ASEBO` | Adaptive Sampling Evolution-Based Optimization | ES |
| `PersistentES` | Persistent Evolution Strategy | ES |
| `NoiseReuseES` | Noise Reuse Evolution Strategy | ES |
| `GuidedES` | Guided Evolution Strategy | ES |
| `ESMC` | Evolution Strategy with Monte Carlo | ES |

## `EvoxPSO` — Multi-Algorithm Wrapper (on `main`)

`EvoxPSO` is a single class that wraps **7 distinct EvoX PSO variants** selectable via its `variant` parameter.

| Variant | Full Name |
|---------|-----------|
| `PSO` | Standard Particle Swarm Optimization |
| `CLPSO` | Comprehensive Learning PSO |
| `CSO` | Competitive Swarm Optimizer |
| `DMSPSOEL` | Dynamic Multi-Swarm PSO with Elite Learning |
| `FSPSO` | Fitness-Sharing PSO |
| `SLPSOGS` | Social Learning PSO with Gaussian Sampling |
| `SLPSOUS` | Social Learning PSO with Uniform Sampling |

## `EvosaxES` — Multi-Algorithm Wrapper (on `algorithm/evo-algorithms`)

`EvosaxES` is a separate evosax-backed wrapper (pure JAX, JIT-friendly) that wraps **14 variants**.
Coexists with `EvoxES` — different backend, class names use `evosax_` prefix.

| Variant | Full Name | Type |
|---------|-----------|------|
| `OpenES` | OpenAI ES (evosax) | ES |
| `XNES` | Exponential NES (evosax) | NES |
| `SNES` | Separable NES (evosax) | NES |
| `ARS` | Augmented Random Search (evosax) | ES |
| `ASEBO` | Adaptive Sampling (evosax) | ES |
| `PersistentES` | Persistent ES (evosax) | ES |
| `NoiseReuseES` | Noise Reuse ES (evosax) | ES |
| `GuidedES` | Guided ES (evosax) | ES |
| `ESMC` | ES with Monte Carlo (evosax) | ES |
| `DES` | Distributed ES (evosax) | ES |
| `PGPE` | Policy Gradients with Parameter Exploration | NES |
| `CR_FM_NES` | Fast Moving NES (evosax) | NES |
| `DE` | Differential Evolution (evosax) | DE |
| `PSO` | Particle Swarm Optimization (evosax) | PSO |

---

## Algorithm Table

### Gradient-Based (Optax / JAX)

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| Adam | `AdamGD` | **Finished** | `main` | Optax Adam |
| NAdam | `NAAdamGD` | **Finished** | `main` | Optax NAdam |
| SA-GD (Simulated Annealing GD) | `SAGD` | **Finished** | `main` | Custom gradient + annealing |
| L-BFGS | `LBFGSGD` | **Finished** | `main` | Optax L-BFGS |
| OptaxGD (generic) | `OptaxGD` | Future branch | `future` | Generic Optax wrapper |
| AdaBelief | `AdaBeliefGD` | Branch | `algorithm/optax-algorithms` | |
| AdaFactor | `AdafactorGD` | Branch | `algorithm/optax-algorithms` | |
| AdamW | `AdamWGD` | Branch | `algorithm/optax-algorithms` | |
| Adan | `AdanGD` | Branch | `algorithm/optax-algorithms` | |
| Lion | `LionGD` | Branch | `algorithm/optax-algorithms` | |
| Lookahead | `LookaheadGD` | Branch | `algorithm/optax-algorithms` | |
| Nadam (Optax) | `NadamGD` | Branch | `algorithm/optax-algorithms` | |
| NAG (Nesterov Accelerated) | `NAG` | Branch | `algorithm/optax-algorithms` | |
| Noisy SGD | `NoisySGD` | Branch | `algorithm/optax-algorithms` | |
| RAdam | `RAdamGD` | Branch | `algorithm/optax-algorithms` | |
| RMSProp | `RMSPropGD` | Branch | `algorithm/optax-algorithms` | |
| SAM (Sharpness-Aware) | `SAM` | Branch | `algorithm/optax-algorithms` | |
| Schedule-Free Adam | `ScheduleFreeAdamGD` | Branch | `algorithm/optax-algorithms` | |
| SGD | `SGD` | Branch | `algorithm/optax-algorithms` | |
| SGDM (SGD + Momentum) | `SGDM` | Branch | `algorithm/optax-algorithms` | |
| Sophia | `SophiaGD` | Branch | `algorithm/optax-algorithms` | |
| Yogi | `YogiGD` | Branch | `algorithm/optax-algorithms` | |

### Gradient-Based (SciPy Wrappers)

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| BFGS | `BFGS` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| L-BFGS-B | `LBFGSB` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| Newton-CG | `NewtonCG` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| Dogleg | `Dogleg` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| TNC | `TNC` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| SLSQP | `SLSQP` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| SR1 | `SR1` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| Trust-Constr | `TrustConstr` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| Trust-Krylov | `TrustKrylov` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| Trust-NCG | `TrustNCG` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| Nonlinear CG | `NonlinearCG` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| COBYLA (SciPy-grad) | `COBYLA` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |
| COBYQA | `COBYQA` | Branch | `algorithm/scipy-grad-algorithms` | Also in `bo-algorithms` |

### Gradient-Based (Custom JAX)

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| SGLD | `SGLDJAX` | Branch | `algorithm/jax-algorithms` | Stochastic Gradient Langevin Dynamics |
| ASAM | `ASAMJAX` | Branch | `algorithm/jax-algorithms` | Adaptive Sharpness-Aware Minimization |
| Adam→L-BFGS Switch | `AdamToLBFGSJAX` | Branch | `algorithm/jax-algorithms` | Hybrid warm-start strategy |
| Entropy-SGD | `EntropySGDJAX` | Branch | `algorithm/jax-algorithms` | |
| SGHMC | `SGHMCJAX` | Branch | `algorithm/jax-algorithms` | Stochastic Gradient Hamiltonian MC |
| ARC | `ARCJAX` | Branch | `algorithm/jax-algorithms` | Adaptive Regularization with Cubics |
| OGD | `OGDJAX` | Branch | `algorithm/jax-algorithms` | Online Gradient Descent |
| OAdam | `OAdamJAX` | Branch | `algorithm/jax-algorithms` | Optimistic Adam |
| Perturbed GD | `PerturbedGDJAX` | Branch | `algorithm/jax-algorithms` | |
| Noisy Adam | `NoisyAdamJAX` | Branch | `algorithm/jax-algorithms` | |
| GD with Restarts | `GDRestartsJAX` | Branch | `algorithm/jax-algorithms` | Periodic restart strategy |
| Gaussian Smoothing GD | `GaussianSmoothingGDJAX` | Branch | `algorithm/jax-algorithms` | Zeroth-order via smoothing |

### Evolutionary

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| EvoX ES (12 variants) | `EvoxES` | **Finished** | `main` | See variant table above |
| EvoX PSO (7 variants) | `EvoxPSO` | **Finished** | `main` | See variant table above |
| Random Search | `RandomSearch` | **Finished** | `main` | Baseline |
| Evosax ES (14 variants) | `EvosaxES` | Branch | `algorithm/evo-algorithms` | See variant table above |
| DE | `DE` | Branch | `algorithm/evo-algorithms` | Differential Evolution (custom impl) |
| JADE | `JADE` | Branch | `algorithm/evo-algorithms` | Adaptive DE |
| SHADE | `SHADE` | Branch | `algorithm/evo-algorithms` | Success-History Adaptive DE |
| L-SHADE | `LSHADE` | Branch | `algorithm/evo-algorithms` | Linear pop. reduction SHADE |
| CoDE | `CoDE` | Branch | `algorithm/evo-algorithms` | Composite DE |
| SaDE | `SaDE` | Branch | `algorithm/evo-algorithms` | Self-adaptive DE |
| jDE | `jDE` | Branch | `algorithm/evo-algorithms` | Self-adaptive DE variant |
| PGPE | `PGPE` | Branch | `algorithm/evo-algorithms` | Policy Gradients with Parameter Exploration |
| CR-FM-NES | `CRFmNES` | Branch | `algorithm/evo-algorithms` | Fast Moving NES |
| CLPSO | `CLPSO` | Branch | `algorithm/evo-algorithms` | Comprehensive Learning PSO |
| FIPS | `FIPS` | Branch | `algorithm/evo-algorithms` | Fully Informed PSO |
| QPSO | `QPSO` | Branch | `algorithm/evo-algorithms` | Quantum-behaved PSO |
| CEM | `CEM` | Branch | `algorithm/evo-algorithms` | Cross-Entropy Method |
| ABC | `ABC` | Branch | `algorithm/evo-algorithms` | Artificial Bee Colony |
| GWO | `GWO` | Branch | `algorithm/evo-algorithms` | Grey Wolf Optimizer |
| IHS | `IHS` | Branch | `algorithm/evo-algorithms` | Improved Harmony Search |
| Nevergrad NGOpt | `NevergradNGOpt` | Branch | `algorithm/nevergrad-algorithms` | |
| Nevergrad OnePlusOne | `NevergradOnePlusOne` | Branch | `algorithm/nevergrad-algorithms` | |
| Nevergrad TBPSA | `NevergradTBPSA` | Branch | `algorithm/nevergrad-algorithms` | |
| CMA-ES | `CMAESCMA` | Branch | `algorithm/cma-es-algorithms` | Via `cmaes` library |
| Sep-CMA-ES | `CMAESSepCMA` | Branch | `algorithm/cma-es-algorithms` | Via `cmaes` library |
| Evosax MA-ES | `EvosaxMAES` | Branch | `algorithm/cma-es-algorithms` | |
| Evosax LM-MA-ES | `EvosaxLMMAES` | Branch | `algorithm/cma-es-algorithms` | Large-scale variant |
| JAX (1+1)-ES | `JAXOnePlusOneES` | Branch | `algorithm/cma-es-algorithms` | Native JAX |
| JAX (μ,λ)-ES | `JAXMuLambdaES` | Branch | `algorithm/cma-es-algorithms` | Native JAX |
| PyCMA CMA-ES | `PyCMACMAES` | Branch | `algorithm/cma-es-algorithms` | Via `pycma` library |
| PyCMA Active CMA-ES | `PyCMAActiveCMAES` | Branch | `algorithm/cma-es-algorithms` | |
| PyCMA IPOP | `PyCMAIPOP` | Branch | `algorithm/cma-es-algorithms` | Increasing-population restart |
| PyCMA BIPOP | `PyCMABIPOP` | Branch | `algorithm/cma-es-algorithms` | Bi-population restart |
| Differential Evolution | `DifferentialEvolution` | Future branch | `future` | |

### Derivative-Free

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| COBYLA | `COBYLA` | Future branch | `future` | SciPy wrapper |
| Nelder-Mead | `NelderMead` | Future branch | `future` | SciPy wrapper |
| Powell | `Powell` | Future branch | `future` | SciPy wrapper |
| PDFO LINCOA | `PDFOLINCOA` | Branch | `algorithm/powell-dfo-algorithms` | Linearly Constrained |
| PDFO NEWUOA | `PDFONEWUOA` | Branch | `algorithm/powell-dfo-algorithms` | Unconstrained |
| PDFO UOBYQA | `PDFOUOBYQA` | Branch | `algorithm/powell-dfo-algorithms` | Unconstrained, quadratic approx |
| Py-BOBYQA | `PyBOBYQA` | Branch | `algorithm/powell-dfo-algorithms` | Bound-constrained |
| Nelder-Mead (scipy-nongrad) | `NelderMead` | Branch | `algorithm/scipy-nongrad-algorithms` | Overlaps with `future` |
| Powell (scipy-nongrad) | `Powell` | Branch | `algorithm/scipy-nongrad-algorithms` | Overlaps with `future` |
| OMADS MADS | `OmadsMADS` | Branch | `algorithm/mads-algorithms` | Mesh Adaptive Direct Search |
| OMADS OrthoMADS | `OmadsOrthoMADS` | Branch | `algorithm/mads-algorithms` | Orthogonal variant |

### Global Search

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| Basin Hopping | `BasinHopping` | Branch | `algorithm/scipy-nongrad-algorithms` | SciPy stochastic global optimizer |
| Dual Annealing | `DualAnnealing` | Branch | `algorithm/scipy-nongrad-algorithms` | SciPy stochastic global optimizer |

### Surrogate-Based / Bayesian Optimization

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| BoTorch BO | `BotorchBO` | **Finished** | `main` | Standard GP-based BO |
| BoTorch TuRBO | `BotorchTuRBO` | **Finished** | `main` | Trust-region BO |
| ReSTIR | `ReSTIR` | **Finished** | `main` | Reservoir Sampling |
| BAxUS | `BAxUS` | Branch | `algorithm/bo-algorithms` | Ax-based, high-dim BO |
| SAASBO | `AxSAASBO` | Branch | `algorithm/bo-algorithms` | Sparse Axis-Aligned Subspace BO |
| GEBO | `GEBO` | Branch | `algorithm/bo-algorithms` | Gradient-Enhanced BO |
| LineBO | `LineBO` | Branch | `algorithm/bo-algorithms` | Line-search BO |
| qKG | `BotorchqKG` | Branch | `algorithm/bo-algorithms` | Knowledge Gradient |
| qNEI | `BotorchqNEI` | Branch | `algorithm/bo-algorithms` | Noisy Expected Improvement |
| REMBO | `REMBO` | Branch | `algorithm/bo-algorithms` | Random Embedding BO |
| HEBO | `HEBO` | Branch | `algorithm/bo-algorithms` | Heteroscedastic BO |
| SMAC | `SMAC` | Branch | `algorithm/bo-algorithms` | Sequential Model-based Config |
| TuRBO-LBFGS | `TuRBOLBFGS` | Branch | `algorithm/bo-algorithms` | TuRBO + L-BFGS hybrid |

### Generative

| Algorithm | Class Name | Status | Branch | Notes |
|-----------|-----------|--------|--------|-------|
| VAE Sampling | `VAESampling` | **Finished** | `main` | Variational Autoencoder |

## Summary by Status

| Status | Count | Categories Covered |
|--------|------:|--------------------|
| **Finished** (on `main`) | 11 classes (22 runnable via variants) | Gradient, Evolutionary, Derivative-Free, Global Search, Surrogate, Generative |
| **Future branch** | 5 | Derivative-Free, Evolutionary, Gradient |
| **Branch** (merge-ready) | 74 classes (88 runnable via variants) | Gradient (Optax, SciPy, JAX), Evolutionary, Derivative-Free, Global Search, Surrogate |
| **Total unique runnable algorithms** | ~115 | 6 categories |

> **Runnable count breakdown:** 11 main classes yield 22 via EvoX variants (12 ES + 7 PSO + 3 standalone).
> Branch classes include EvosaxES with 14 additional variants.

---

## Branch Merge Priority

All branches below merge cleanly with `main` (0 conflicts verified).

| Priority | Branch | New Algorithms | Notes |
|----------|--------|---------------:|-------|
| HIGH | `algorithm/optax-algorithms` | 17 | |
| HIGH | `algorithm/evo-algorithms` | 17 + EvosaxES (14 variants) | |
| HIGH | `algorithm/scipy-grad-algorithms` | 13 | |
| MEDIUM | `algorithm/powell-dfo-algorithms` | 4 | |
| MEDIUM | `algorithm/jax-algorithms` | 12 | |
| MEDIUM | `algorithm/nevergrad-algorithms` | 3 | |
| MEDIUM | `algorithm/cma-es-algorithms` | 9 | Evosax overlap with evo-algorithms |
| MEDIUM | `algorithm/bo-algorithms` | 10 | SciPy grad overlap with `scipy-grad-algorithms` (near-identical; merge scipy-grad first) |
| MEDIUM | `algorithm/scipy-nongrad-algorithms` | 4 (2 unique) | Nelder-Mead & Powell overlap with `future` |
| LOW | `algorithm/mads-algorithms` | 2 | Niche but unique paradigm |
