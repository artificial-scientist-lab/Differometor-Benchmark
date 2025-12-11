# Notes
## Short-Term TODO
- ~~Add outputs at the end of an algorithm to continue later on~~
  > Added the option to input starting parameters
- ~~Fix adam to start over for the next optimize() call~~
  > Implemented a way of initializing its parameters
- ~~Find out how to initialize PSO population~~
- ~~Add mean/std to benchmarking (and to typst table)~~
- ~~Implement more precise time tracking~~
  > Implemented `wall_time_indices`
- ~~Implement the precise time tracking in `benchmark.py`~~
- Implement a class that handles all evox algorithms
  - Derive many different optimizers from this one
- (?) Remove the behaviour of PSO when `n_generations` and `wall_times` is set
- Fix AUC drift

## Tmp notes
- Standard Deviations werden gebenchmarked
- Kein Padding von losses unterschiedlicher runs mehr
- Runs können insgesamt gesaved und geladen werden
  - `best_params_history` speichern?
- Für alle Algorithmen
  - Akzeptieren `random_seed` für Sachen wie `torch.seed`
  - Akzeptieren `init_params`
  - Akzeptieren `wall_times`
  - Geben `wall_time_indices` zurück
- Adam fixes
- Runtime type checking
- Added bayesian optimization
  - Zu langsam?
- Brauchen wir das feature für `max_iterations` + `wall_times`?

## Refactoring Notes
- Class docstrings for attributes

## Notes
- Eine gute Initialisierung immer als richtlinie währed optimierung
- Stabilität bei Initialisierung charakterisieren
- Man kann noch eininges viel modularer machen! Aber erstmal verschiedene Algos ausprobieren von EvoX
- UMAP um setups mit Eigenschaften zu Lösungen
- Speichern von Lösungen!

## Metric notes

### Sören
- Siehe Notes

### PMO Benchmark
- AUC of top-K average property value vs. calls (min-max scaled to [0,1])

### Tartarus
- Average of best fitness (+std)
- Initialization time
- Time per call
- Total wall clock time (if limited budget of calls)
- Solution Diversity (Euclid distance)
- (Rate of success, Time to success)

### GuacaMol / Moses
- Solution Diversity
- (Validity: Rate of success)
- Rest not applicable (has to do with input data)
- Maybe for Voyager: Similarity to voyager parameters
- MOSES: Similarity to Nearest Neighbor

### Dockstring
- (Calls to sucess, loss as a fx of call)
- (other irrelevant metrics for our case)

## Benchmark notes

- Random Seed TODO
  - Need to adapt algorithm implementations
- Runs that complete fewer iterations in the same time are getting edge padded (assuming convergence)
- If wall_time is given to adam, any iteration limits are currently ignored.
  - Would this be needed?
  - It's implemented in PSO
- Cool plots could be a good idea for Benchmark results.
- Why was AUC sinking in the first runs?

## Modularization changelog
I am modularizing voyager_optimization to prepare for other algorithms and hyperparameter optimization. This should later be a subclass of a uifo optmization class which modularizes the more general optimization of a uifo setup. I think that starting with modularizing this more simple optimization is better.
### HOW TO USE
Here's an example of how to run the new modulraized code with fixed hyper parameters:
```python
vp = VoyagerProblem()

optimizer = EvoxPSO(problem=vp)

optimizer.optimize(pop_size=50, n_generations=10)
```
### Moved everything into `./optimization/`
As there will be many optimization algorithms, this was necessary. Created subfolders
- `./optimization/algorithms/`
- `./optimization/voyager/` for applications of 
### Created `class VoyagerProblem`
- Moved much of the setup into `__init__()`
- Properties are:
  -  `optimization_pairs`
  -  `n_params` for convenience, is equal to `len(optimization_pairs)`.
- Methods of the class:
  - `objective_function(optimized_parameters)` as a regular public attribute so it is pure and callable (wihtout self-reference and not as a property).
    - This function gets defined in `__init__()` to maintain pureness (otherwise it would use mutable `self.` variables). TODO review if this is necessary. 
  - `t2j(a)` for array conversion between `torch` and `jax`
  - `j2t(a)`
  - `output_to_files()` see below. TODO better name
- Variables are (crossed ones were removed as a class variable):
  - `self._name`: Used for file naming. TODO Should this as an argumentent maybe be moved to `output_to_files()`?
  - `self._setup`: Needed for `update_setup()` at file output.
  - `self._frequencies`: Needed for `df.run()` at file output
  - ~~`self._carrier`~~
  - ~~`self._signal`~~
  - ~~`self._noise`~~
  - ~~`self._detector_ports`~~
  - `self._target_sensitivity`
  - ~~`self._target_loss`~~
  - `self._optimization_pairs`: Needed for optimization algorithm
  - ~~`self._simulation_arrays`~~
  - ~~`self._detector_ports`~~
  - `self._bounds`: Needed for `update_setup()` for file output.
- File output is a new function `output_to_files()`
  - Outputs  best paramters and losses (over time) into json
  - Outputs Loss curve over time TODO option to plot over function calls.
  - Outputs the final setup's sensitivity curve.
  - Each argument can be left out which will just not plot/output the corresponding data.
  - Maybe looking for some magic methods
  - TODO check whether `hyper_param_str` for example is empty and do sth
  - Arguments are:
    - `best_params`: Params for the final setup that get printed
    - `losses`: Array of losses. Decided on arrays for Loss tracking because the other input is also an array. Lists of losses in algorithms will have to be converted. Some algorithms may need to use lists internally because of the "dynamic" stopping criterion. 
    - `population_losses`: 2D Array of losses for each pop. Graph could get messy. TODO: find a good plotting representation for high population cases. TODO better name!
    - `algorithm_str`
    - `hyper_param_str`
    - `hyper_param_str_in_filename`
### Created `ContinuousProblem(ABC)` as a base class / interface of `VoyagerProblem`
- Functions and attributes:
- `objective_function: Callable[[Float[Array, "{self.n_params}"]], Float]` (attribute!)
- `__init__(self, name: String)`
- `optimization_pairs(self) -> list[tuple]`
- Implemented `n_params(self) -> int`
- Implemented `t2j(a: Array) -> jax.Array`
- Implemented `j2t(a: Array) -> torch.Tensor`
- `output_to_files(output_to_files( self, best_params: Float[Array, "{self.n_params}"], losses: Float[Array, "iterations"], population_losses: Float[Array, "iterations pop"], algorithm_str: str, hyper_param_str: str, hyper_param_str_in_filename: bool) -> None`
### Created `OptimizationAlgorithm(ABC)`
- Some base class to implement algorithms
- Taking in a Problem and hyperparameters and solving
- Hyperparam optimization should then happen if you run `optimize()` with different params multiple times
- Attributes:
  - `algorithm_str: str`
  - `__init__(self, problem: ContinuousProblem, *args, **kwargs)`
  - `optimize(self, save_to_file: bool, *args, **kwargs)`
### Created `AdamGD` and `EvoxPSO` as subclasses of `OptimizationAlgorithm`
- Implemented the algorithms accordingly


### Added `jaxtyping`
- Using type annotation like `Float[Array, "iterations pop"]

## Candidate Libraries
### PyGAD
- not that great
- Slow because of CPU
### EvoX (maybe also EvoGP, EvoRL)
- Seem great but are torch based
- Older jax-based EvalX version (0.9.0) exists

### evosax
- Seems good
- jax-based
- many algorithms (no PSO seemingly)

### numpyro
- jax-based, relies on jax' autodiff
- For bayesian optimization

## Algos to try

### Gradient stuff
- Annealing!
- PSO gradient based?

### Particle Swarm Optimization (PSO)
### Resources
- [Getting started with PSO (EvoX)](https://evox.readthedocs.io/en/latest/examples/so-algorithm.html)
- [Custom Problems in EvoX](https://evox.readthedocs.io/en/stable/tutorial/tutorial_part5.html)

### Differential Evolution (DE)

### Bayesian Optimization
- Optimization is performing poorly in the beginning
  - Maybe this will not be the case later on
- 


-  `bayesian-optimization` package is way too slow. Need another one
- Candidates are:
  - BoTorch
  - GPJax (with Optax)
    - Probably better if objective fun were less complex
  - Boax
    - Jax-based contender to BoTorch?