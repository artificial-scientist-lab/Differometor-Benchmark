import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from differometor.utils import sigmoid_bounding
from dfbench.core.problem import ContinuousProblem
from dfbench.core.display import LiveDisplay, LogDisplay
from dfbench.core.storage import (
    CheckpointManager,
    CheckpointSerializer,
    JsonCheckpointSerializer,
    NpzCheckpointSerializer,
    RunDataExporter,
    RunMetadata,
    RunPathResolver,
    RunState,
    SaveConfig,
    StorageBackend,
    LocalFilesystemBackend,
)


class Objective:
    """Instrumented wrapper around a ContinuousProblem for benchmarking optimizers.

    Objective acts as the sole interface between an optimization algorithm
    and the underlying problem.  It forwards every function evaluation through
    JAX while transparently recording losses, gradients, parameters, and
    wall-clock timestamps so that different algorithms can be compared on a
    fair, reproducible basis.

    Core responsibilities
    ---------------------
    1. **Function evaluation** – exposes ``value``, ``grad``, ``hessian``,
       ``value_and_grad``, and ``value_grad_and_hessian`` for single-point
       queries as well as ``vmap_value``, ``vmap_grad``, ``vmap_hessian``,
       and their combined variants (aliased as ``batched_*``) for batched
       evaluation.  The instance is also callable: ``obj(params)`` is
       equivalent to ``obj.value(params)``.

    2. **Budget enforcement** – honours ``max_evals`` and ``max_time``
       constraints.  Once a budget is exhausted the ``budget_exceeded``
       flag is raised and further evaluations are no longer logged.

    3. **History tracking** – maintains aligned histories of losses, params,
       gradients, evaluation types, and elapsed time.  The ``save`` list of
       string tokens (plus the two standard flags ``save_time_steps`` and
       ``save_params_history``) controls what is stored; the active
       configuration is recorded as a :class:`SaveConfig` and embedded in
       every checkpoint so a resumed run can detect mismatches.

        4. **Bounded / unbounded mode** – when ``unbounded=True`` the objective
             is evaluated through a configurable mapping from unbounded to bounded
             space (default: sigmoid transform) so that algorithms can optimise in
             unconstrained $(-\\infty, +\\infty)$ space while the underlying
             problem remains bounded.  Custom mappings map to the [0, 1] range;
             the Objective scales to actual bounds automatically.

    5. **Reproducible random sampling** – ``set_seed`` initialises a JAX
       PRNG key that is consumed by ``random_params_bounded`` and
       ``random_params_unbounded``, guaranteeing identical initial
       populations across runs.

    6. **Checkpointing & I/O** – ``save_run_data`` / ``load_run_data``
       persist the full optimisation state to compressed NPZ files;
       ``output_to_files`` writes human-readable JSON + PNG summaries.

    Typical usage
    -------------
    >>> from dfbench.core.problem import ContinuousProblem
    >>> problem = ContinuousProblem(...)
    >>> obj = Objective(problem, max_evals=5000, max_time=60.0)
    >>> obj.set_seed(42)
    >>> obj.start_logging()
    >>> while not obj.budget_exceeded:
    ...     params = obj.random_params_bounded()
    ...     loss, grad = obj.value_and_grad(params)
    ...     # ... update params with your algorithm ...
    >>> print(obj.best_loss, obj.best_params_bounded)

    Public methods
    --------------
    **Evaluation**

    - ``value(params)``            – scalar loss.
    - ``grad(params)``             – gradient vector.
    - ``hessian(params)``          – Hessian matrix.
    - ``value_and_grad(params)``   – both in one forward+backward pass.
    - ``value_grad_and_hessian(params)`` – loss, gradient, and Hessian.
    - ``vmap_value(params)``       – batched losses   (alias ``batched_value``).
    - ``vmap_grad(params)``        – batched gradients (alias ``batched_grad``).
    - ``vmap_hessian(params)``     – batched Hessians (alias ``batched_hessian``).
    - ``vmap_value_and_grad(params)`` – batched loss + grad.
    - ``vmap_value_grad_and_hessian(params)`` – batched loss + grad + Hessian.

    **Lifecycle**

    - ``warmup_*()``               – deterministic two-call JAX warmups; call
      before ``start_logging()``.
    - ``start_logging()``          – starts the wall-clock timer; call before optimising.
    - ``reset()``                  – clears all histories / counters for a fresh run.

    **Random sampling**

        - ``set_seed(seed)``             – set JAX PRNG seed for reproducibility.
        - ``random_params(n)``           – samples from the active parameter space.
        - ``random_params_bounded(n)``   – uniform samples inside parameter bounds.
        - ``random_params_unbounded(n)`` – samples mapped to unbounded space via
            inverse mapping (default: inverse-sigmoid).  For custom mappings
            the Objective normalises to [0, 1] before calling the inverse.

    **I/O**

    - ``save_run_data(...)``       – checkpoint to compressed NPZ.
    - ``load_run_data(filepath)``  – restore from checkpoint.
    - ``output_to_files(...)``     – write JSON params/losses + PNG plots.
    - ``get_summary()``            – dict snapshot of current run statistics.

    Key properties
    --------------
    +------------------------------------+---------------------------------------------------+
    | Property                           | Description                                       |
    +====================================+===================================================+
    | ``bounds``                         | ``(2, n_params)`` lower/upper bounds              |
    |                                    | (or unbounded).                                   |
    +------------------------------------+---------------------------------------------------+
    | ``n_params``                       | Number of optimisable parameters.                 |
    +------------------------------------+---------------------------------------------------+
    | ``problem``                        | The wrapped ``ContinuousProblem``.                |
    +------------------------------------+---------------------------------------------------+
    | ``eval_count``                     | Total evaluations so far.                         |
    +------------------------------------+---------------------------------------------------+
    | ``evals_left``                     | Remaining evaluation budget (``None`` if          |
    |                                    | unlimited).                                       |
    +------------------------------------+---------------------------------------------------+
    | ``evals_exceeded``                 | Whether the evaluation budget is exhausted.       |
    +------------------------------------+---------------------------------------------------+
    | ``time_left``                      | Remaining seconds (``None`` if unlimited).        |
    +------------------------------------+---------------------------------------------------+
    | ``time_elapsed``                   | Seconds since ``start_logging()``.                |
    +------------------------------------+---------------------------------------------------+
    | ``time_exceeded``                  | Whether the time budget is exhausted.             |
    +------------------------------------+---------------------------------------------------+
    | ``budget_exceeded``                | ``True`` when *any* budget is exhausted.          |
    +------------------------------------+---------------------------------------------------+
    | ``best_loss``                      | Lowest loss observed (``None`` before first       |
    |                                    | eval).                                            |
    +------------------------------------+---------------------------------------------------+
    | ``best_params``                    | Raw params at ``best_loss`` (may be               |
    |                                    | unbounded).                                       |
    +------------------------------------+---------------------------------------------------+
    | ``best_params_bounded``            | Best params mapped back to bounded space.         |
    +------------------------------------+---------------------------------------------------+
    | ``current_loss``                   | Loss from the most recent evaluation.             |
    +------------------------------------+---------------------------------------------------+
    | ``current_params``                 | Params from the most recent evaluation.           |
    +------------------------------------+---------------------------------------------------+
    | ``loss_history``                   | List of all recorded losses (copy).               |
    +------------------------------------+---------------------------------------------------+
    | ``grad_history``                   | List of all recorded gradients (copy).            |
    +------------------------------------+---------------------------------------------------+
    | ``hessian_history``                | List of all recorded Hessians (copy).             |
    +------------------------------------+---------------------------------------------------+
    | ``params_history``                 | List of all recorded params (copy, raw).          |
    +------------------------------------+---------------------------------------------------+
    | ``params_history_bounded``         | Params history mapped to bounded space.           |
    +------------------------------------+---------------------------------------------------+
    | ``time_steps``                     | Elapsed-time stamps aligned with histories.      |
    +------------------------------------+---------------------------------------------------+
    | ``improvement_count``              | Times ``best_loss`` was improved.                 |
    +------------------------------------+---------------------------------------------------+
    | ``evals_since_improvement``        | Evaluations since last improvement.               |
    +------------------------------------+---------------------------------------------------+
    | ``evals_progress_fraction``        | Fraction of eval budget consumed (0–1).           |
    +------------------------------------+---------------------------------------------------+
    | ``time_progress_fraction``         | Fraction of time budget consumed (0–1).           |
    +------------------------------------+---------------------------------------------------+
    | ``budget_left_fraction``           | Fraction of tightest budget remaining (1->0).     |
    +------------------------------------+---------------------------------------------------+
    | ``budget_progress_fraction``       | Fraction of tightest budget consumed (0->1).      |
    +------------------------------------+---------------------------------------------------+
    | ``loss_history_reduced``           | Losses with batches reduced to min.               |
    +------------------------------------+---------------------------------------------------+
    | ``params_history_reduced``         | Params with batches reduced to single entry.      |
    +------------------------------------+---------------------------------------------------+
    | ``params_history_reduced_bounded`` | Reduced params in bounded space.                  |
    +------------------------------------+---------------------------------------------------+
    | ``grad_history_reduced``           | Grads with batches reduced to single entry.       |
    +------------------------------------+---------------------------------------------------+
    | ``hessian_history_reduced``        | Hessians with batches reduced to single entry.    |
    +------------------------------------+---------------------------------------------------+

    Notes
    -----
    - All ``*_reduced`` properties collapse batched entries to a single
      representative value (argmin of loss, then argmin of gradient norm,
      then argmin of Hessian norm, then first element).
    - ``jax.grad``, ``jax.hessian``, ``jax.value_and_grad``, and ``jax.vmap``
      variants are prepared up front, so warmup is recommended before
      timing-sensitive runs.
    - Checkpoints are saved atomically (write to a temp file in the same
      directory, then :func:`os.replace`) by the configured
      :class:`~dfbench.core.storage.StorageBackend` to prevent corruption
      from interrupted jobs.
    """

    def __init__(
        self,
        problem: ContinuousProblem,
        unbounded: bool = False,
        max_evals: int | None = None,
        max_time: float | None = None,
        save_time_steps: bool = True,
        save_params_history: bool = True,
        save: list[str] | None = None,
        verbose: int = 0,
        print_every: int = 100,
        algorithm_str: str | None = None,
        save_to_file_every: int | None = None,
        display_mode: str = "live",
        unit_mapping: Callable | None = None,
        inverse_unit_mapping: Callable | None = None,
        hessian_batch_size: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_dir: str | Path | None = None,
    ):
        """Initialize the Objective wrapper for optimization problems.

        Args:
            problem: The continuous optimization problem to wrap.
            unbounded: If True, use unbounded objective mode with the active
                mapping (default: sigmoid). Defaults to False.
            max_evals: Maximum number of evaluations allowed. None for unlimited.
            max_time: Maximum wall-clock time in seconds. None for unlimited.
            save_time_steps: Whether to track timestamps for each evaluation.
            save_params_history: Whether to save parameter history.
            save: List of advanced save tokens for recording additional /
                batched histories. Valid tokens: ``"grad"``, ``"hessian"``,
                ``"eval_type"``, ``"batched_loss"``, ``"batched_grad"``,
                ``"batched_hessian"``, ``"batched_param"``, ``"batched"``
                (convenience alias expanding to all four batched tokens).
                Defaults to ``None`` (no advanced histories).
            verbose: Verbosity level (0=silent, 1=warnings, 2=info). Defaults to 0.
            print_every: Print progress every N evaluations (if verbose >= 1). Defaults to 100.
            algorithm_str: String identifier for the optimization algorithm.
            save_to_file_every: Save checkpoint every N evaluations. None to
                disable. The time spent saving is excluded from the
                elapsed-time clock.
            display_mode: How to display progress when ``verbose >= 1``.
                ``"live"`` (default) shows a continuously-refreshing in-place
                dashboard with progress bars.  ``"log"`` prints traditional
                multi-line log blocks that scroll the terminal.
            unit_mapping: Optional function that maps unbounded
                parameters to the **[0, 1] range** (unit interval).  Can be a
                scalar function (e.g. ``jax.nn.sigmoid``) or a vector function
                operating element-wise on arrays — both work because JAX
                broadcasts element-wise operations.  The Objective handles
                scaling from [0, 1] to the actual problem bounds:
                ``bounded = lower + (upper - lower) * f(unbounded)``.
                Must be provided together with ``inverse_unit_mapping``.
            inverse_unit_mapping: Inverse of
                ``unit_mapping``, mapping from [0, 1] back to
                unbounded space.  The Objective normalises bounded parameters
                to [0, 1] before calling this function:
                ``unbounded = f_inv((bounded - lower) / (upper - lower))``.
                Must be provided together with ``unit_mapping``.
            hessian_batch_size: Number of Hessian columns to compute
                simultaneously via ``vmap``.  Higher values trade GPU memory
                for speed.  ``1`` (default) is the most memory-efficient
                (sequential ``lax.map``); set to ``n_params`` to recover
                full ``jax.hessian`` parallelism.  Values between 1 and
                ``n_params`` compute columns in chunks.
            checkpoint_format: On-disk format for checkpoints. ``"npz"``
                (default) writes compressed NumPy archives; ``"json"`` writes a
                pickle-free, human-readable JSON file — useful when loading
                checkpoints from untrusted sources or when you want to inspect
                them by hand. No extra imports needed.
            checkpoint_dir: Root directory for checkpoint and output artifacts.
                Defaults to ``./data/objective_run_data``. Pass a path to
                redirect all artifacts (e.g. to a scratch disk or a
                ``tmp_path`` in tests) without importing any storage class.

        To customise the storage stack beyond these two knobs (e.g. a custom
        serializer or a non-filesystem backend), subclass :class:`Objective`
        and override :meth:`_build_storage`.
        """

        self.unbounded = unbounded
        self.algorithm_str = algorithm_str
        self._problem = problem
        self._max_time = max_time
        self._max_evals = max_evals
        self._print_every = print_every
        self._verbose = verbose
        self._save_config = SaveConfig.from_flags(
            save_time_steps=save_time_steps,
            save_params_history=save_params_history,
            save=save,
        )
        self._display_mode = display_mode
        self._hessian_batch_size = hessian_batch_size

        self._set_space_mappings(
            unit_mapping,
            inverse_unit_mapping,
        )

        self._bounds = problem.bounds
        self._bind_evaluation_functions()

        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._start_time = None
        self._time_offset = 0.0

        # --- Modular storage ----------------------------------------------
        # Assembled by _build_storage from the user-facing checkpoint_format
        # and checkpoint_dir knobs. Subclasses override _build_storage to swap
        # a custom serializer / backend / resolver / exporter.
        self._resolver: RunPathResolver
        self._serializer: CheckpointSerializer
        self._backend: StorageBackend
        self._exporter: RunDataExporter
        self._checkpoint_manager: CheckpointManager
        self._build_storage(checkpoint_format, checkpoint_dir, save_to_file_every)

        self._eval_count = 0
        self._evals_left = self._max_evals
        self._evals_exceeded = False

        self._best_loss = jnp.inf
        self._improvement_count = 0
        self._evals_since_improvement = 0
        self._best_params = None
        self._loss_history = []
        self._grad_history = []
        self._hessian_history = []
        self._params_history = []
        self._eval_type_history = []
        self._time_steps = []

        # Random seed for reproducibility (set by algorithm via set_seed method)
        self._seed = None
        self._rng_key = None

        # Lightweight call-type tracking (always active, O(1) per call)
        self._log_call_count: int = 0
        self._eval_type_counts: dict[int, int] = {}

        # Display renderer (lazy-initialised on first use)
        self._display: LiveDisplay | LogDisplay | None = None

    def _set_space_mappings(
        self,
        unit_mapping: Callable | None,
        inverse_unit_mapping: Callable | None,
    ) -> None:
        """Set or clear custom [0,1]-space mappings.

        The forward mapping must produce values in [0, 1]; the Objective
        handles scaling to actual bounds.  The inverse mapping receives
        values already normalised to [0, 1] by the Objective.
        """
        if (unit_mapping is None) != (inverse_unit_mapping is None):
            raise ValueError(
                "Custom unbounded mapping requires both "
                "unit_mapping and inverse_unit_mapping."
            )

        self._unit_mapping = unit_mapping
        self._inverse_unit_mapping = inverse_unit_mapping
        self._unit_mapping_vmap = (
            jax.vmap(unit_mapping) if unit_mapping is not None else None
        )
        self._inverse_unit_mapping_vmap = (
            jax.vmap(inverse_unit_mapping) if inverse_unit_mapping is not None else None
        )

    # ------------------------------------------------------------------
    # Storage assembly
    # ------------------------------------------------------------------

    _SERIALIZERS: dict[str, CheckpointSerializer] = {
        "npz": NpzCheckpointSerializer(),
        "json": JsonCheckpointSerializer(),
    }

    def _build_storage(
        self,
        checkpoint_format: str,
        checkpoint_dir: str | Path | None,
        save_every: int | None,
    ) -> None:
        """Assemble the storage stack from user-facing knobs.

        Maps ``checkpoint_format`` to a serializer, roots the resolver and
        backend at ``checkpoint_dir`` (defaulting to the historical
        ``./data/objective_run_data``), and wires the
        :class:`CheckpointManager`. Subclasses override this to swap a custom
        serializer / backend / resolver / exporter.
        """
        fmt = checkpoint_format.lower()
        if fmt not in self._SERIALIZERS:
            raise ValueError(
                f"Unknown checkpoint_format '{checkpoint_format}'. "
                f"Valid formats: {sorted(self._SERIALIZERS)}."
            )
        self._checkpoint_format = fmt
        self._checkpoint_dir = checkpoint_dir
        self._serializer = self._SERIALIZERS[fmt]
        self._resolver = RunPathResolver(
            root=(
                str(checkpoint_dir)
                if checkpoint_dir is not None
                else "./data/objective_run_data"
            )
        )
        self._backend = LocalFilesystemBackend(root=self._resolver.root)
        self._exporter = RunDataExporter()
        self._checkpoint_manager = CheckpointManager(
            backend=self._backend,
            serializer=self._serializer,
            resolver=self._resolver,
            save_every=save_every,
        )

    def _map_unbounded_to_bounded(
        self,
        params: Float[Array, "n_params"],
    ) -> Float[Array, "n_params"]:
        """Map a single parameter vector from unbounded to bounded space.

        Custom mappings produce [0, 1] values; this method scales to bounds.
        """
        if self._unit_mapping is not None:
            lower, upper = self._problem.bounds
            return lower + (upper - lower) * self._unit_mapping(params)
        return sigmoid_bounding(params, self._problem.bounds)

    def _map_unbounded_to_bounded_batched(
        self,
        params: Float[Array, "batch n_params"],
    ) -> Float[Array, "batch n_params"]:
        """Map a parameter batch from unbounded to bounded space."""
        if self._unit_mapping_vmap is not None:
            lower, upper = self._problem.bounds
            return lower + (upper - lower) * self._unit_mapping_vmap(params)
        return jax.vmap(lambda x: sigmoid_bounding(x, self._problem.bounds))(params)

    def _map_bounded_to_unbounded(
        self,
        params: Float[Array, "... n_params"],
    ) -> Float[Array, "... n_params"]:
        """Map params from bounded to unbounded space using configured inverse.

        For custom mappings, normalises to [0, 1] first, then calls the
        user-provided inverse which maps [0, 1] → (-∞, +∞).
        """
        if self._inverse_unit_mapping is not None:
            arr = jnp.asarray(params)
            lower, upper = self._problem.bounds
            normalized = (arr - lower) / (upper - lower)
            if arr.ndim == 1:
                return self._inverse_unit_mapping(normalized)
            assert self._inverse_unit_mapping_vmap is not None
            return self._inverse_unit_mapping_vmap(normalized)
        return self._inverse_sigmoid_bounding(params, self._problem.bounds)

    def value_function(self, *, unbounded: bool | None = None) -> Callable:
        """Return an unlogged JAX-compatible scalar value function.

        Args:
            unbounded: If True, map unbounded params to bounded space before
                calling the problem objective. If False, call the problem
                objective directly. None uses the Objective's active mode.
        """
        use_unbounded = self.unbounded if unbounded is None else unbounded
        if use_unbounded:

            def _unbounded_value(params):
                bounded = self._map_unbounded_to_bounded(params)
                return self._problem.objective_function(bounded)

            return _unbounded_value
        return self._problem.objective_function

    def _bind_evaluation_functions(self) -> None:
        """Bind evaluation callables for the currently active search space."""
        self._func = self.value_function()
        self._grad_func = jax.jit(jax.grad(self._func))
        self._value_and_grad_func = jax.jit(jax.value_and_grad(self._func))

        # Memory-efficient Hessian: compute columns in chunks via
        # forward-over-reverse (jvp of grad).  hessian_batch_size controls
        # how many columns are computed in parallel (1 = fully sequential,
        # n_params = fully parallel like jax.hessian).
        _grad_for_hessian = jax.grad(self._func)
        _hbs = self._hessian_batch_size

        def _batched_hessian(params):
            n = params.shape[0]

            def _cols_chunk(basis_chunk):
                """Compute multiple Hessian columns in parallel."""

                def _single_col(e_i):
                    _, col = jax.jvp(_grad_for_hessian, (params,), (e_i,))
                    return col

                return jax.vmap(_single_col)(basis_chunk)

            # Build full identity and split into chunks
            basis = jnp.eye(n)
            # Pad to multiple of batch size for lax.map
            remainder = n % _hbs
            if remainder != 0:
                pad_size = _hbs - remainder
                basis = jnp.concatenate([basis, jnp.zeros((pad_size, n))], axis=0)
            chunks = basis.reshape(-1, _hbs, n)
            # lax.map iterates sequentially over chunks
            result = jax.lax.map(_cols_chunk, chunks)
            # Reshape back: (n_chunks, batch_size, n_params) -> (total, n_params)
            result = result.reshape(-1, n)
            return result[:n]  # Remove padding rows

        self._hessian_func = _batched_hessian

        def _value_grad_and_hessian(params):
            value, grad = self._value_and_grad_func(params)
            hessian = self._hessian_func(params)
            return value, grad, hessian

        self._value_grad_and_hessian_func = _value_grad_and_hessian
        self._vmap_func = jax.vmap(self._func)
        self._vmap_grad_func = jax.vmap(self._grad_func)
        self._vmap_hessian_func = jax.vmap(self._hessian_func)
        self._vmap_value_and_grad_func = jax.vmap(self._value_and_grad_func)
        self._vmap_value_grad_and_hessian_func = jax.vmap(
            self._value_grad_and_hessian_func
        )

    def set_space_mode(
        self,
        unbounded: bool,
        unit_mapping: Callable | None = None,
        inverse_unit_mapping: Callable | None = None,
    ) -> None:
        """Switch between bounded and unbounded evaluation mode.

        This rebinds all internal JAX callables so subsequent ``value*`` /
        ``grad*`` / ``hessian*`` evaluations use the requested objective.

        Args:
            unbounded: If True, evaluate in unbounded mode using the active
                mapping. If False, use ``problem.objective_function`` directly.
            unit_mapping: Optional function mapping unbounded
                parameters to the [0, 1] range.  Can be scalar (e.g.
                ``jax.nn.sigmoid``) or element-wise vector.  The Objective
                scales to actual bounds: ``lb + (ub - lb) * f(x)``.
                Must be passed together with ``inverse_unit_mapping``.
            inverse_unit_mapping: Inverse of the forward mapping,
                mapping [0, 1] → (-∞, +∞).  The Objective normalises bounded
                params to [0, 1] before calling this.  Must be passed together
                with ``unit_mapping``.

        Raises:
            RuntimeError: If logging already started for the current run.
        """
        if self._start_time is not None:
            raise RuntimeError(
                "set_space_mode() must be called before start_logging() so "
                "evaluation histories stay consistent."
            )
        if (unit_mapping is None) != (inverse_unit_mapping is None):
            raise ValueError(
                "set_space_mode() custom mapping requires both "
                "unit_mapping and inverse_unit_mapping."
            )
        if unit_mapping is not None:
            self._set_space_mappings(
                unit_mapping,
                inverse_unit_mapping,
            )
        self.unbounded = unbounded
        self._bind_evaluation_functions()

    def __call__(self, params: Float[Array, "n_params"]) -> Float:
        """Evaluate the objective function at given parameters.

        Args:
            params: Parameter vector to evaluate.

        Returns:
            Loss value at the given parameters.
        """
        return self.value(params)

    # --------- Problem Information Properties ---------

    @property
    def bounds(self) -> Float[Array, "2 n_params"] | None:
        """Lower and upper bounds for parameters as shape (2, n_params) array."""
        return self._bounds

    @property
    def n_params(self) -> int:
        """Number of parameters in the optimization problem."""
        if self._bounds is not None:
            return self.problem.n_params
        else:
            raise ValueError("Cannot determine n_params for unbounded objective.")

    @property
    def problem(self) -> ContinuousProblem:
        """The underlying optimization problem."""
        return self._problem

    # --------- Optimization Tracking Functions/Properties ---------

    @property
    def eval_count(self) -> int:
        """Total number of objective evaluations performed."""
        return self._eval_count

    @property
    def max_evals(self) -> int | None:
        """The evaluation budget, or ``None`` if unlimited."""
        return self._max_evals

    @property
    def max_time(self) -> float | None:
        """The wall-clock time budget in seconds, or ``None`` if unlimited."""
        return self._max_time

    @property
    def evals_left(self) -> int | None:
        """Number of evaluations remaining before budget is exceeded. None if no limit."""
        return self._evals_left

    @property
    def evals_progress_fraction(self) -> float:
        """Fraction of evaluation budget used (0.0 to 1.0). Returns 0.0 if no limit."""
        if self._max_evals is not None:
            return min(1.0, self.eval_count / self._max_evals)
        return 0.0

    @property
    def evals_exceeded(self) -> bool:
        """Whether the evaluation budget has been exceeded."""
        return self._evals_exceeded

    @property
    def time_left(self) -> float | None:
        """Time remaining in seconds before budget is exceeded. None if no limit."""
        if self._max_time is None:
            return None
        if self._start_time is None:
            return max(0.0, self._max_time - self._time_offset)
        return max(0.0, self._max_time - self.time_elapsed)

    @property
    def time_elapsed(self) -> float:
        """Total time elapsed (including any previously loaded offset)."""
        if self._start_time is None:
            return self._time_offset
        return time.time() - self._start_time

    @property
    def time_exceeded(self) -> bool:
        """Whether the time budget has been exceeded."""
        if self._max_time is None:
            return False
        return self.time_elapsed >= self._max_time

    @property
    def time_progress_fraction(self) -> float:
        """Fraction of time budget used (0.0 to 1.0). Returns 0.0 if no limit."""
        if self._max_time is not None:
            return min(1.0, self.time_elapsed / self._max_time)
        return 0.0

    @property
    def budget_progress_fraction(self) -> float:
        """Fraction of the tightest budget consumed (0.0 -> 1.0).

        Computed as ``max(evals_progress_fraction, time_progress_fraction)``
        considering only the budgets that are actually set.
        Returns 0.0 when no budget is configured.
        """
        fracs: list[float] = []
        if self._max_evals is not None:
            fracs.append(self.evals_progress_fraction)
        if self._max_time is not None:
            fracs.append(self.time_progress_fraction)
        if not fracs:
            return 0.0
        return min(1.0, max(fracs))

    @property
    def budget_left_fraction(self) -> float:
        """Fraction of the tightest budget remaining (1.0 -> 0.0).

        Equivalent to ``1 - budget_progress_fraction``.
        Returns 1.0 when no budget is configured.
        """
        return 1.0 - self.budget_progress_fraction

    @property
    def budget_exceeded(self) -> bool:
        """Whether any budget (time or evaluations) has been exceeded."""
        return self.time_exceeded or self.evals_exceeded

    @property
    def best_params(self) -> Float[Array, "n_params"] | None:
        """Parameters corresponding to best loss found so far (raw, possibly unbounded)."""
        return self._best_params

    @property
    def best_params_bounded(self) -> Float[Array, "n_params"] | None:
        """Best parameters transformed to bounded space. Use for final output."""
        if self._best_params is None:
            return None
        if self.unbounded:
            return self._map_unbounded_to_bounded(self._best_params)
        return self._best_params

    @property
    def best_loss(self) -> Float | None:
        """Best (minimum) loss found so far. None if no evaluations yet."""
        return self._best_loss if self._best_loss != jnp.inf else None

    @property
    def current_loss(self) -> Float[Array, "batch"] | Float | None:
        """Most recent loss value from last evaluation."""
        if len(self._loss_history) == 0:
            return None
        return self._loss_history[-1]

    @property
    def current_params(
        self,
    ) -> Float[Array, "batch n_params"] | Float[Array, "n_params"] | None:
        """Most recent parameters from last evaluation."""
        if len(self._params_history) > 0:
            return self._params_history[-1]
        else:
            return None

    @property
    def loss_history(self) -> list[Float | Float[Array, "batch"]]:
        """Copy of all loss values computed (prevents external modification)."""
        return self._loss_history.copy()

    @property
    def grad_history(self) -> list[Float | Float[Array, "batch"]]:
        """Copy of all gradient values computed (prevents external modification)."""
        return self._grad_history.copy()

    @property
    def hessian_history(
        self,
    ) -> list[
        Float[Array, "n_params n_params"]
        | Float[Array, "batch n_params n_params"]
        | None
    ]:
        """Copy of all Hessian values computed (prevents external modification)."""
        return self._hessian_history.copy()

    @property
    def params_history(
        self,
    ) -> list[Float[Array, "n_params"] | Float[Array, "batch n_params"]]:
        """Copy of all parameter values evaluated (raw, possibly unbounded)."""
        return self._params_history.copy()

    @property
    def params_history_bounded(
        self,
    ) -> list[Float[Array, "n_params"] | Float[Array, "batch n_params"] | None]:
        """Params history transformed to bounded space. Use for final output/plotting."""
        if not self.unbounded:
            return self._params_history.copy()
        result = []
        for p in self._params_history:
            if p is None:
                result.append(None)
            elif p.ndim == 1:
                result.append(self._map_unbounded_to_bounded(p))
            else:
                result.append(self._map_unbounded_to_bounded_batched(p))
        return result

    @property
    def time_steps(self) -> list[float]:
        """Copy of elapsed time at each evaluation in seconds."""
        return self._time_steps.copy()

    @property
    def improvement_count(self) -> int:
        """Number of times a new best loss was found."""
        return int(self._improvement_count)

    @property
    def evals_since_improvement(self) -> int:
        """Evaluations since last improvement to best loss."""
        return int(self._evals_since_improvement)

    @property
    def log_call_count(self) -> int:
        """Total number of internal _log_evals() invocations (not evaluations).

        Unlike ``eval_count`` which counts individual parameter evaluations,
        this counts how many times logging was triggered, making it possible
        to derive the average batch size::

            avg_batch = obj.eval_count / obj.log_call_count
        """
        return self._log_call_count

    @property
    def eval_type_counts(self) -> dict[int, int]:
        """Distribution of evaluation call types as ``{type_code: count}`` dict.

        Type codes are bitmasks: ``0b{hess}{vmap}{grad}{loss}``

        +------+-----------------------------+
        | Code | Meaning                     |
        +======+=============================+
        |  1   | value only                  |
        +------+-----------------------------+
        |  2   | grad only                   |
        +------+-----------------------------+
        |  3   | value + grad                |
        +------+-----------------------------+
        |  8   | hessian only                |
        +------+-----------------------------+
        | 11   | value + grad + hessian      |
        +------+-----------------------------+
        |  5   | batched value               |
        +------+-----------------------------+
        |  6   | batched grad                |
        +------+-----------------------------+
        |  7   | batched value + grad        |
        +------+-----------------------------+
        | 12   | batched hessian             |
        +------+-----------------------------+
        | 15   | batched value + grad + hess |
        +------+-----------------------------+
        | -1   | unknown (params only)       |
        +------+-----------------------------+
        """
        return dict(self._eval_type_counts)

    @property
    def last_checkpoint_eval(self) -> int | None:
        """Eval count at which the most recent checkpoint was written, or None."""
        return self._checkpoint_manager.last_checkpoint_eval

    @property
    def save_config(self) -> SaveConfig:
        """The :class:`SaveConfig` describing which histories are recorded."""
        return self._save_config

    @property
    def save_every(self) -> int | None:
        """Periodic checkpoint cadence in evaluations, or ``None`` if disabled."""
        return self._checkpoint_manager.save_every

    @property
    def checkpoint_format(self) -> str:
        """On-disk checkpoint format (``"npz"`` or ``"json"``)."""
        return self._checkpoint_format

    @property
    def checkpoint_dir(self) -> str | Path | None:
        """Root directory for checkpoint artifacts, or ``None`` for the default."""
        return self._checkpoint_dir

    # --------- Reduced (non-batched) history properties ---------

    @property
    def loss_history_reduced(self) -> list[float]:
        """Loss history with batches reduced to min.

        Always returns a list of scalar floats, regardless of whether
        batched losses were saved. For batched entries, returns nanmin.
        """
        result = []
        for entry in self._loss_history:
            arr = jnp.asarray(entry)
            if arr.ndim == 0:
                result.append(float(arr))
            else:
                result.append(float(jnp.nanmin(arr)))
        return result

    @property
    def params_history_reduced(
        self,
    ) -> list[Float[Array, "n_params"] | None]:
        """Params history with batches reduced to single representative.

        Always returns a list of 1D param arrays (or None), regardless of
        whether batched params were saved. For batched entries, selects:
        1. Params with minimum loss (if loss available for that step)
        2. Params with smallest gradient norm (if grad available)
        3. Params with smallest Hessian norm (if Hessian available)
        4. First entry in batch (fallback)
        """
        result = []
        for i, params in enumerate(self._params_history):
            if params is None:
                result.append(None)
                continue

            params_arr = jnp.asarray(params)
            if params_arr.ndim == 1:
                # Already scalar params
                result.append(params_arr)
            else:
                idx = self._representative_index(
                    loss=self._loss_history[i] if i < len(self._loss_history) else None,
                    grad=(
                        self._grad_history[i] if i < len(self._grad_history) else None
                    ),
                    hessian=(
                        self._hessian_history[i]
                        if i < len(self._hessian_history)
                        else None
                    ),
                )
                result.append(params_arr[idx])
        return result

    @property
    def params_history_reduced_bounded(
        self,
    ) -> list[Float[Array, "n_params"] | None]:
        """Reduced params history transformed to bounded space.

        Combines params_history_reduced with bounding transformation.
        Use this for final output, plotting, or benchmark analysis.
        """
        reduced = self.params_history_reduced
        if not self.unbounded:
            return reduced
        result = []
        for p in reduced:
            if p is None:
                result.append(None)
            else:
                result.append(self._map_unbounded_to_bounded(p))
        return result

    @property
    def grad_history_reduced(
        self,
    ) -> list[Float[Array, "n_params"] | None]:
        """Grad history with batches reduced to single representative.

        Always returns a list of 1D grad arrays (or None). For batched entries,
        selects using the same logic as params_history_reduced.
        """
        result = []
        for i, grad in enumerate(self._grad_history):
            if grad is None:
                result.append(None)
                continue

            grad_arr = jnp.asarray(grad)
            if grad_arr.ndim == 1:
                result.append(grad_arr)
            else:
                idx = self._representative_index(
                    loss=self._loss_history[i] if i < len(self._loss_history) else None,
                    grad=grad_arr,
                    hessian=(
                        self._hessian_history[i]
                        if i < len(self._hessian_history)
                        else None
                    ),
                )
                result.append(grad_arr[idx])
        return result

    @property
    def hessian_history_reduced(
        self,
    ) -> list[Float[Array, "n_params n_params"] | None]:
        """Hessian history with batches reduced to a single representative."""
        result = []
        for i, hessian in enumerate(self._hessian_history):
            if hessian is None:
                result.append(None)
                continue

            hessian_arr = jnp.asarray(hessian)
            if hessian_arr.ndim == 2:
                result.append(hessian_arr)
            else:
                idx = self._representative_index(
                    loss=self._loss_history[i] if i < len(self._loss_history) else None,
                    grad=(
                        self._grad_history[i] if i < len(self._grad_history) else None
                    ),
                    hessian=hessian_arr,
                )
                result.append(hessian_arr[idx])
        return result

    # --------- Random seed management ---------

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible parameter sampling.

        This method should be called by optimization algorithms to ensure
        reproducibility across different runs. The seed is used to initialize
        a JAX random key that is automatically consumed and updated each time
        random_params_bounded() or random_params_unbounded() is called without
        an explicit rng_key argument.

        Args:
            seed: Integer seed for random number generation.

        Example:
            >>> obj = Objective(problem)
            >>> obj.set_seed(42)  # Called by algorithm
            >>> params1 = obj.random_params_bounded(n_samples=100)
            >>> params2 = obj.random_params_bounded(n_samples=100)  # Different samples
            >>> obj.set_seed(42)  # Reset
            >>> params3 = obj.random_params_bounded(n_samples=100)  # Same as params1
        """
        self._seed = seed
        self._rng_key = jax.random.PRNGKey(seed)

    # --------- Random parameter sampling ---------

    def random_params(
        self,
        n_samples: int = 1,
        rng_key=None,
    ) -> Float[Array, "n_samples n_params"] | Float[Array, "n_params"]:
        """Generate random parameters in the active objective space.

        Returns unbounded samples when ``self.unbounded`` is True, otherwise
        returns bounded samples. This is the preferred helper for algorithms
        that should follow the space selected by ``prepare()`` / ``set_space_mode()``.

        Args:
            n_samples: Number of parameter vectors to generate. Defaults to 1.
            rng_key: Optional JAX random key. If None, uses the Objective's
                internal key when available.

        Returns:
            Array of shape (n_samples, n_params) if n_samples > 1,
            or (n_params,) if n_samples == 1.
        """
        if self.unbounded:
            return self.random_params_unbounded(n_samples=n_samples, rng_key=rng_key)
        return self.random_params_bounded(n_samples=n_samples, rng_key=rng_key)

    def random_params_bounded(
        self,
        n_samples: int = 1,
        rng_key=None,
    ) -> Float[Array, "n_samples n_params"] | Float[Array, "n_params"]:
        """Generate random parameters in bounded space.

        Samples uniformly within the parameter bounds.

        Reproducibility:
            - If rng_key is provided: Uses that specific key (manual control)
            - If rng_key is None and set_seed() was called: Uses internal key
              (automatically split/updated for each call)
            - Otherwise: Falls back to numpy random (non-reproducible)

        Args:
            n_samples: Number of parameter vectors to generate. Defaults to 1.
            rng_key: JAX random key for manual control. If None, uses internal
                key set by set_seed(), or falls back to numpy random.

        Returns:
            Array of shape (n_samples, n_params) if n_samples > 1,
            or (n_params,) if n_samples == 1.

        Example:
            >>> obj = Objective(problem)
            >>> obj.set_seed(42)  # Set by algorithm
            >>> samples = obj.random_params_bounded(n_samples=1000)
            >>> # Or with manual key:
            >>> key = jax.random.PRNGKey(123)
            >>> samples = obj.random_params_bounded(n_samples=1000, rng_key=key)
        """
        if self._bounds is None:
            raise ValueError(
                "Cannot sample bounded params: bounds are None (unbounded objective)."
            )

        lower, upper = self._bounds[0], self._bounds[1]

        # Determine which random key to use
        if rng_key is not None:
            # Use provided key (manual override)
            key_to_use = rng_key
        elif self._rng_key is not None:
            # Use internal key and split it for next call
            key_to_use, self._rng_key = jax.random.split(self._rng_key)
        else:
            key_to_use = None

        if key_to_use is not None:
            # Use JAX random
            samples = jax.random.uniform(
                key_to_use,
                shape=(n_samples, self.n_params),
                minval=lower,
                maxval=upper,
            )
        else:
            # Fallback to numpy random (non-reproducible)
            samples = np.random.uniform(
                low=lower,
                high=upper,
                size=(n_samples, self.n_params),
            )
            samples = jnp.asarray(samples)

        # Return 1D if single sample
        if n_samples == 1:
            return samples[0]
        return samples

    def random_params_unbounded(
        self,
        n_samples: int = 1,
        rng_key=None,
    ) -> Float[Array, "n_samples n_params"] | Float[Array, "n_params"]:
        """Generate random parameters in unbounded space.

        Samples uniformly in bounded space, then applies the configured
        inverse mapping (default: inverse sigmoid / logit) to map to
        unbounded space (-∞, +∞).

        The inverse mapping ensures that when these unbounded params are
        passed through the matching forward mapping, they recover the
        original bounded samples.

        Reproducibility:
            - If rng_key is provided: Uses that specific key (manual control)
            - If rng_key is None and set_seed() was called: Uses internal key
            - Otherwise: Falls back to numpy random (non-reproducible)

        Args:
            n_samples: Number of parameter vectors to generate. Defaults to 1.
            rng_key: JAX random key for manual control. If None, uses internal
                key set by set_seed(), or falls back to numpy random.

        Returns:
            Array of shape (n_samples, n_params) if n_samples > 1,
            or (n_params,) if n_samples == 1.

        Example:
            >>> obj = Objective(problem, unbounded=True)
            >>> obj.set_seed(42)  # Set by algorithm for reproducibility
            >>> samples = obj.random_params_unbounded(n_samples=1000)
        """
        # Generate bounded samples (will use internal key if set)
        bounded_samples = self.random_params_bounded(n_samples, rng_key=rng_key)

        # Apply configured inverse mapping (default: inverse sigmoid)
        unbounded = self._map_bounded_to_unbounded(bounded_samples)

        return unbounded

    @staticmethod
    def _inverse_sigmoid_bounding(
        bounded_params: Float[Array, "... n_params"],
        bounds: Float[Array, "2 n_params"],
    ) -> Float[Array, "... n_params"]:
        """Inverse of sigmoid_bounding: maps bounded params to unbounded space.

        Given bounded parameters in [lower, upper], computes unbounded parameters
        in (-∞, +∞) such that sigmoid_bounding(unbounded, bounds) = bounded.

        The sigmoid bounding formula is:
            bounded = lower + (upper - lower) * sigmoid(unbounded)
        where sigmoid(x) = 1 / (1 + exp(-x))

        The inverse is:
            unbounded = logit((bounded - lower) / (upper - lower))
        where logit(p) = log(p / (1 - p))

        Args:
            bounded_params: Parameters in bounded space.
            bounds: Array of shape (2, n_params) with [lower_bounds, upper_bounds].

        Returns:
            Parameters in unbounded space.
        """
        lower, upper = bounds[0], bounds[1]

        # Normalize to [0, 1]
        normalized = (bounded_params - lower) / (upper - lower)

        # Clip to prevent numerical issues with logit at boundaries
        # logit(0) = -inf, logit(1) = +inf
        eps = 1e-7
        normalized = jnp.clip(normalized, eps, 1.0 - eps)

        # Apply logit transform: logit(p) = log(p / (1 - p))
        unbounded = jnp.log(normalized / (1.0 - normalized))

        return unbounded

    def _deterministic_warmup_params(
        self,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_params"]:
        """Return deterministic midpoint params in the currently active raw space."""
        if self._bounds is None:
            raise ValueError(
                "Cannot create deterministic warmup params without finite bounds."
            )
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")

        midpoint = (self._bounds[0] + self._bounds[1]) / 2.0
        bounded_params = (
            midpoint[None, :]
            if n_samples == 1
            else jnp.repeat(midpoint[None, :], repeats=n_samples, axis=0)
        )

        if self.unbounded:
            return self._map_bounded_to_unbounded(bounded_params)
        return bounded_params

    def _warmup_twice(self, fn, params) -> None:
        """Execute a deterministic warmup twice before logging begins."""
        if self._start_time is not None:
            raise RuntimeError(
                "warmup_*() must be called before start_logging() to avoid "
                "affecting budgets and histories."
            )
        fn(params)
        fn(params)

    # ---------

    def _get_run_data_path(
        self,
        algorithm_name: str = "unknown",
        custom_path: str | None = None,
        hyper_param_str: str | None = None,
    ) -> Path:
        """Generate run data file path via the configured path resolver.

        Delegates to :class:`~dfbench.core.storage.RunPathResolver` so the
        path layout is not hardcoded here.

        Args:
            algorithm_name: Name of the optimization algorithm.
            custom_path: Custom path to override default. If None, uses the
                resolver's structured layout.
            hyper_param_str: Optional hyperparameter string for subdirectory
                organization.

        Returns:
            Path object for the run data file.
        """
        if custom_path is not None:
            return Path(custom_path)

        problem_name = (
            self._problem.name if hasattr(self._problem, "name") else "problem"
        )
        path = self._resolver.checkpoint_path(
            problem_name=problem_name,
            algorithm_name=algorithm_name,
            timestamp=self._timestamp,
            hyper_param_str=hyper_param_str,
            max_time=self._max_time,
            max_evals=self._max_evals,
        )
        return path

    def __repr__(self) -> str:
        """String representation for debugging."""
        summary = self.get_summary()
        best_loss = summary["best_loss"]
        best_loss_str = f"{best_loss:.6f}" if best_loss is not None else "N/A"
        return (
            f"Objective(evals={summary['eval_count']}, "
            f"best_loss={best_loss_str}, "
            f"time={summary['time_elapsed']:.2f}s)"
        )

    # --------- Display rendering ---------

    def _ensure_display(self) -> None:
        """Lazily create the display renderer if not yet initialised."""
        if self._display is not None:
            return
        if self._display_mode == "live":
            self._display = LiveDisplay(self)
        else:
            self._display = LogDisplay(self)

    def _render_display(self) -> None:
        """Render one frame of the progress display (live or log)."""
        self._ensure_display()
        assert self._display is not None
        self._display.render()

    def finalize_display(self) -> None:
        """Print a final, non-overwritable summary after the optimisation run.

        Call this once after the optimisation loop ends to leave a
        persistent status block in the terminal.  If ``verbose < 1``
        or no display was created, this is a no-op.
        """
        if self._verbose < 1:
            return
        self._ensure_display()
        assert self._display is not None
        self._display.finalize()

    # --------- internal logging and tracking methods ---------

    @staticmethod
    def _ndim(x) -> int:
        """Helper to get ndim attribute, returning 0 if not present."""
        return getattr(x, "ndim", 0)

    @staticmethod
    def _nanargmin_or_none(values) -> int | None:
        """Return nanargmin index or None if the input is empty / all-NaN."""
        arr = jnp.asarray(values)
        if arr.size == 0 or jnp.all(jnp.isnan(arr)):
            return None
        return int(jnp.nanargmin(arr))

    def _representative_index(self, loss=None, grad=None, hessian=None) -> int:
        """Pick a representative batch index using loss, then grad, then Hessian."""
        if loss is not None and self._ndim(loss) > 0:
            idx = self._nanargmin_or_none(loss)
            if idx is not None:
                return idx

        if grad is not None and self._ndim(grad) > 1:
            grad_norms = jnp.linalg.norm(jnp.asarray(grad), axis=-1)
            idx = self._nanargmin_or_none(grad_norms)
            if idx is not None:
                return idx

        if hessian is not None and self._ndim(hessian) > 2:
            flat_hessian = jnp.reshape(jnp.asarray(hessian), (hessian.shape[0], -1))
            hessian_norms = jnp.linalg.norm(flat_hessian, axis=-1)
            idx = self._nanargmin_or_none(hessian_norms)
            if idx is not None:
                return idx

        return 0

    def _log(
        self,
        params: Float[Array, "n_params"] | Float[Array, "batch n_params"] | None = None,
        loss: Float | Float[Array, "batch"] | None = None,
        grad: Float | Float[Array, "batch"] | None = None,
        hessian: Float[Array, "n_params n_params"]
        | Float[Array, "batch n_params n_params"]
        | None = None,
    ) -> None:
        """Internal: Log timestamp, evaluation results, and optionally save to file."""
        if self._start_time is None:
            return
        time_exceeded = self.time_exceeded
        if (
            self._save_config.time_steps
            and not time_exceeded
            and not self._evals_exceeded
        ):
            self._time_steps.append(self.time_elapsed)
        self._log_evals(params, loss, grad, hessian, time_exceeded=time_exceeded)
        self._log_to_file()

    def _log_evals(
        self,
        params: Float[Array, "n_params"] | Float[Array, "batch n_params"] | None = None,
        loss: Float | Float[Array, "batch"] | None = None,
        grad: Float | Float[Array, "batch"] | None = None,
        hessian: Float[Array, "n_params n_params"]
        | Float[Array, "batch n_params n_params"]
        | None = None,
        time_exceeded: bool = False,
    ) -> None:
        """Internal: Log evaluation results, update histories, and track best loss.

        This function is defensive: `params`, `loss`, `grad`, or `hessian` may
        be None.
        Histories are kept index-aligned by inserting NaN placeholders when a
        particular quantity is not provided. An `_eval_types` entry is appended
        describing the kind of evaluation ('value', 'grad', 'value_and_grad').
        """
        # Stop if logging didn't start yet
        if self._start_time is None:
            return

        # Stop logging if budget exceeded
        if time_exceeded or self._evals_exceeded:
            return

        # Determine how many items this call represents
        if params is not None and self._ndim(params) == 2:
            n_items = int(params.shape[0])
        elif loss is not None and self._ndim(loss) > 0:
            n_items = int(loss.shape[0])
        elif grad is not None and self._ndim(grad) > 1:
            n_items = int(grad.shape[0])
        elif hessian is not None and self._ndim(hessian) > 2:
            n_items = int(hessian.shape[0])
        else:
            n_items = 1

        # Check evaluation budget with knowledge of batch size
        if self._max_evals is not None:
            evals_left_before = max(0, self._max_evals - self._eval_count)

            # nothing left before this call: mark exceeded and bail
            if evals_left_before <= 0:
                self._evals_exceeded = True
                self._evals_left = 0
                # Remove the time step that was just added by _log_time() to keep alignment
                if self._save_config.time_steps and self._time_steps:
                    self._time_steps.pop()
                return

            # batch larger than remaining budget: account evals but do not log
            if evals_left_before < n_items:
                # still account for the evaluations (the caller received results),
                # but do not record histories for fairness.
                self._eval_count += n_items
                self._evals_left = max(0, self._max_evals - self._eval_count)
                self._evals_exceeded = True
                # Remove the time step that was just added by _log_time() to keep alignment
                if self._save_config.time_steps and self._time_steps:
                    self._time_steps.pop()
                return

        prev_eval_count = self._eval_count
        self._eval_count += n_items
        # update remaining evals immediately
        if self._max_evals is not None:
            self._evals_left = max(0, self._max_evals - self._eval_count)
            self._evals_exceeded = self._evals_left <= 0
        # Decide eval type
        # Format as bitmask: 0b{hess}{vmap}{grad}{loss}
        if loss is None and grad is None and hessian is None:
            eval_type = -1
        else:
            eval_type = (
                int(hessian is not None) << 3
                | int(n_items > 1) << 2
                | int(grad is not None) << 1
                | int(loss is not None)
            )
        if self._save_config.eval_type:
            self._eval_type_history.append(eval_type)
        # Always track call count and type distribution (O(1), no allocation)
        self._log_call_count += 1
        self._eval_type_counts[eval_type] = self._eval_type_counts.get(eval_type, 0) + 1

        # Helper to create NaN placeholders
        def _nan_entry():
            return jnp.full((n_items,), jnp.nan) if n_items > 1 else jnp.nan

        # log losses
        if loss is not None:
            if self._ndim(loss) == 0 or self._save_config.batched_loss:
                self._loss_history.append(loss)
            else:  # batched case but not saving batched history -> store min
                self._loss_history.append(jnp.nanmin(loss))
        else:
            # insert NaN(s) to keep alignment
            self._loss_history.append(
                jnp.array([jnp.nan] * n_items)
                if (n_items > 1 and self._save_config.batched_loss)
                else _nan_entry()
            )

        # log grads (only when saving grads)
        if self._save_config.grad:
            if grad is not None:
                if self._ndim(grad) == 1 or self._save_config.batched_grad:
                    self._grad_history.append(grad)
                else:
                    idx = self._representative_index(
                        loss=loss, grad=grad, hessian=hessian
                    )
                    self._grad_history.append(grad[idx])
            else:
                self._grad_history.append(None)

        # log Hessians (only when saving Hessians)
        if self._save_config.hessian:
            if hessian is not None:
                if self._ndim(hessian) == 2 or self._save_config.batched_hessian:
                    self._hessian_history.append(hessian)
                else:
                    idx = self._representative_index(
                        loss=loss, grad=grad, hessian=hessian
                    )
                    self._hessian_history.append(hessian[idx])
            else:
                self._hessian_history.append(None)

        # params history (store raw params; use *_bounded properties for bounded access)
        if self._save_config.params:
            if params is not None:
                if self._ndim(params) == 1 or self._save_config.batched_param:
                    self._params_history.append(params)
                else:  # batched case but not saving batched history
                    idx = self._representative_index(
                        loss=loss, grad=grad, hessian=hessian
                    )
                    self._params_history.append(params[idx])
            else:
                # No params provided; append None to keep alignment
                self._params_history.append(None)

        # Update best loss and params (only when loss available)
        improved = False
        if loss is not None:
            if self._ndim(loss) == 0:
                if not jnp.isnan(loss) and loss < self._best_loss:
                    self._best_loss = loss
                    self._best_params = params
                    improved = True
            else:  # batched saved case
                if not jnp.all(jnp.isnan(loss)):
                    min_idx = int(jnp.nanargmin(loss))
                    min_loss = loss[min_idx]
                    if min_loss < self._best_loss:
                        self._best_loss = min_loss
                        # only set best_params if params provided
                        if params is not None:
                            self._best_params = params[min_idx]
                        improved = True

        # Update incremental improvement / stagnation counters
        if improved:
            self._improvement_count += 1
            self._evals_since_improvement = 0
        else:
            # any evaluation that did not improve increments stagnation
            self._evals_since_improvement += n_items

        # Print progress if configured
        if (
            self._print_every is not None
            and self._print_every > 0
            and self._verbose >= 1
            and (prev_eval_count // self._print_every)
            != (self._eval_count // self._print_every)
        ):
            try:
                self._render_display()
            except Exception:
                # printing should not break optimization
                pass

        return

    def _log_to_file(self) -> None:
        """Internal: checkpoint via :meth:`CheckpointManager.tick`.

        The manager owns the cadence (``save_every``) and the save-timing;
        it returns the wall-clock duration of the save so the Objective can
        exclude it from the elapsed-time clock.
        """
        if self._start_time is None:
            return

        dt = self._checkpoint_manager.tick(self._eval_count, self._build_run_state)
        if dt > 0:
            self._start_time += dt

    # --------- public API for optimization ---------

    def warmup_value(self) -> None:
        """Warm up ``value()`` twice on deterministic params without logging."""
        self._warmup_twice(self.value, self._deterministic_warmup_params()[0])

    def warmup_grad(self) -> None:
        """Warm up ``grad()`` twice on deterministic params without logging."""
        self._warmup_twice(self.grad, self._deterministic_warmup_params()[0])

    def warmup_hessian(self) -> None:
        """Warm up ``hessian()`` twice on deterministic params without logging."""
        self._warmup_twice(self.hessian, self._deterministic_warmup_params()[0])

    def warmup_value_and_grad(self) -> None:
        """Warm up ``value_and_grad()`` twice on deterministic params."""
        self._warmup_twice(self.value_and_grad, self._deterministic_warmup_params()[0])

    def warmup_value_grad_and_hessian(self) -> None:
        """Warm up ``value_grad_and_hessian()`` twice on deterministic params."""
        self._warmup_twice(
            self.value_grad_and_hessian,
            self._deterministic_warmup_params()[0],
        )

    def warmup_vmap_value(self, batch_size: int = 2) -> None:
        """Warm up ``vmap_value()`` twice on a deterministic batch.

        Args:
            batch_size: Number of samples in the warmup batch. Defaults to 2.
        """
        self._warmup_twice(
            self.vmap_value,
            self._deterministic_warmup_params(n_samples=batch_size),
        )

    def warmup_vmap_grad(self, batch_size: int = 2) -> None:
        """Warm up ``vmap_grad()`` twice on a deterministic batch.

        Args:
            batch_size: Number of samples in the warmup batch. Defaults to 2.
        """
        self._warmup_twice(
            self.vmap_grad,
            self._deterministic_warmup_params(n_samples=batch_size),
        )

    def warmup_vmap_hessian(self, batch_size: int = 2) -> None:
        """Warm up ``vmap_hessian()`` twice on a deterministic batch.

        Args:
            batch_size: Number of samples in the warmup batch. Defaults to 2.
        """
        self._warmup_twice(
            self.vmap_hessian,
            self._deterministic_warmup_params(n_samples=batch_size),
        )

    def warmup_vmap_value_and_grad(self, batch_size: int = 2) -> None:
        """Warm up ``vmap_value_and_grad()`` twice on a deterministic batch.

        Args:
            batch_size: Number of samples in the warmup batch. Defaults to 2.
        """
        self._warmup_twice(
            self.vmap_value_and_grad,
            self._deterministic_warmup_params(n_samples=batch_size),
        )

    def warmup_vmap_value_grad_and_hessian(self, batch_size: int = 2) -> None:
        """Warm up ``vmap_value_grad_and_hessian()`` twice on a deterministic batch.

        Args:
            batch_size: Number of samples in the warmup batch. Defaults to 2.
        """
        self._warmup_twice(
            self.vmap_value_grad_and_hessian,
            self._deterministic_warmup_params(n_samples=batch_size),
        )

    def start_logging(self) -> None:
        """Start the optimization timer. Call this before beginning optimization.

        If a checkpoint was loaded via ``load_run_data()``, the previously
        elapsed time (stored in ``_time_offset``) is absorbed into
        ``_start_time`` so that ``time_elapsed`` remains the single source of truth.
        """
        self._start_time = time.time() - self._time_offset
        self._time_offset = 0.0

    def log_evaluation(
        self,
        params: Float[Array, "n_params"] | Float[Array, "batch n_params"] | None = None,
        loss: Float | Float[Array, "batch"] | None = None,
        grad: Float[Array, "n_params"] | Float[Array, "batch n_params"] | None = None,
        hessian: Float[Array, "n_params n_params"]
        | Float[Array, "batch n_params n_params"]
        | None = None,
    ) -> None:
        """Manually log an evaluation result. Used for custom evaluation loops which
        should be jitted.

        This method allows external code to log evaluations that may not go through
        the standard value/grad methods. It accepts the same parameters as _log_evals
        and will update histories and best loss accordingly.

        Args:
            params: Parameters evaluated (raw, possibly unbounded).
            loss: Loss value(s) computed.
            grad: Gradient value(s) computed.
            hessian: Hessian value(s) computed.
        """
        self._log(params, loss, grad, hessian)
        return

    def value(self, params: Float[Array, "n_params"]) -> Float:
        """Evaluate objective function at given parameters.

        Args:
            params: Parameter vector of shape (n_params,).

        Returns:
            Scalar loss value.
        """
        loss = self._func(params)

        self._log(params, loss)
        return loss

    def grad(self, params: Float[Array, "n_params"]) -> Float[Array, "n_params"]:
        """Compute gradient of objective function at given parameters.

        Args:
            params: Parameter vector of shape (n_params,).

        Returns:
            Gradient vector of shape (n_params,).
        """
        grad = self._grad_func(params)

        self._log(params, grad=grad)
        return grad

    def hessian(
        self, params: Float[Array, "n_params"]
    ) -> Float[Array, "n_params n_params"]:
        """Compute the Hessian of the objective function at given parameters."""
        hessian = self._hessian_func(params)

        self._log(params, hessian=hessian)
        return hessian

    def value_and_grad(
        self, params: Float[Array, "n_params"]
    ) -> tuple[Float, Float[Array, "n_params"]]:
        """Compute both value and gradient (more efficient than separate calls).

        Args:
            params: Parameter vector of shape (n_params,).

        Returns:
            Tuple of (loss, gradient).
        """
        value, grad = self._value_and_grad_func(params)

        self._log(params, value, grad)
        return value, grad

    def value_grad_and_hessian(
        self, params: Float[Array, "n_params"]
    ) -> tuple[Float, Float[Array, "n_params"], Float[Array, "n_params n_params"]]:
        """Compute value, gradient, and Hessian at a single parameter vector."""
        value, grad, hessian = self._value_grad_and_hessian_func(params)

        self._log(params, value, grad, hessian)
        return value, grad, hessian

    def vmap_value(
        self, params: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch"]:
        """Evaluate objective function on a batch of parameters.

        Args:
            params: Parameter batch of shape (batch, n_params).

        Returns:
            Loss array of shape (batch,).
        """
        losses = self._vmap_func(params)

        self._log(params, losses)
        return losses

    def vmap_grad(
        self, params: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch n_params"]:
        """Compute gradients for a batch of parameters.

        Args:
            params: Parameter batch of shape (batch, n_params).

        Returns:
            Gradient array of shape (batch, n_params).
        """
        grads = self._vmap_grad_func(params)

        self._log(params, grad=grads)
        return grads

    def vmap_hessian(
        self, params: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch n_params n_params"]:
        """Compute Hessians for a batch of parameters."""
        hessians = self._vmap_hessian_func(params)

        self._log(params, hessian=hessians)
        return hessians

    def vmap_value_and_grad(
        self, params: Float[Array, "batch n_params"]
    ) -> tuple[Float[Array, "batch"], Float[Array, "batch n_params"]]:
        """Compute both values and gradients for a batch of parameters.

        Args:
            params: Parameter batch of shape (batch, n_params).

        Returns:
            Tuple of (losses, gradients) with shapes (batch,) and (batch, n_params).
        """
        values, grads = self._vmap_value_and_grad_func(params)

        self._log(params, values, grads)
        return values, grads

    def vmap_value_grad_and_hessian(
        self, params: Float[Array, "batch n_params"]
    ) -> tuple[
        Float[Array, "batch"],
        Float[Array, "batch n_params"],
        Float[Array, "batch n_params n_params"],
    ]:
        """Compute values, gradients, and Hessians for a batch of parameters."""
        values, grads, hessians = self._vmap_value_grad_and_hessian_func(params)

        self._log(params, values, grads, hessians)
        return values, grads, hessians

    def batched_value(
        self, params: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch"]:
        """Alias for vmap_value. Evaluate objective on a batch of parameters."""
        return self.vmap_value(params)

    def batched_grad(
        self, params: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch n_params"]:
        """Alias for vmap_grad. Compute gradients for a batch of parameters."""
        return self.vmap_grad(params)

    def batched_hessian(
        self, params: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch n_params n_params"]:
        """Alias for vmap_hessian. Compute Hessians for a batch of parameters."""
        return self.vmap_hessian(params)

    def batched_value_and_grad(
        self, params: Float[Array, "batch n_params"]
    ) -> tuple[Float[Array, "batch"], Float[Array, "batch n_params"]]:
        """Alias for vmap_value_and_grad. Compute values and gradients for a batch."""
        return self.vmap_value_and_grad(params)

    def batched_value_grad_and_hessian(
        self, params: Float[Array, "batch n_params"]
    ) -> tuple[
        Float[Array, "batch"],
        Float[Array, "batch n_params"],
        Float[Array, "batch n_params n_params"],
    ]:
        """Alias for vmap_value_grad_and_hessian on a batch of parameters."""
        return self.vmap_value_grad_and_hessian(params)

    # Redirect everythging else to jax. ...probably a bad idea
    # def __getattr__(self, name: str) -> Callable:
    #     if hasattr(jax, name):
    #         return getattr(jax, name)
    #     raise AttributeError(f"'Objective' object has no attribute '{name}'")

    # --------- public API for I/O ---------

    def _build_metadata(self, algorithm_name: str | None = None) -> RunMetadata:
        """Build a :class:`RunMetadata` snapshot for the current run.

        If the wrapped problem implements the reconstructive
        :meth:`~dfbench.core.problem.ContinuousProblem.to_spec` contract,
        its spec is embedded in ``metadata.extra["problem_spec"]`` so the
        checkpoint fully describes which problem instance produced the run.
        """
        extra: dict[str, Any] = {"save_config": self._save_config.to_dict()}
        spec_fn = getattr(self._problem, "to_problem_spec", None)
        if callable(spec_fn):
            try:
                extra["problem_spec"] = spec_fn().to_dict()
            except Exception:
                # Fall back to the legacy to_spec() dict if the typed
                # container is unavailable. A problem that fails to
                # describe itself should not break checkpointing; the run
                # is still saveable, just not self-reconstructing.
                legacy = getattr(self._problem, "to_spec", None)
                if callable(legacy):
                    try:
                        extra["problem_spec"] = legacy()
                    except Exception:
                        pass
        else:
            legacy = getattr(self._problem, "to_spec", None)
            if callable(legacy):
                try:
                    extra["problem_spec"] = legacy()
                except Exception:
                    pass
        return RunMetadata(
            problem_name=(
                self._problem.name if hasattr(self._problem, "name") else "problem"
            ),
            algorithm_name=algorithm_name or self.algorithm_str or "unknown",
            hyper_param_str="",
            timestamp=self._timestamp,
            max_time=self._max_time,
            max_evals=self._max_evals,
            unbounded=self.unbounded,
            extra=extra,
        )

    def _build_run_state(self, algorithm_name: str | None = None) -> RunState:
        """Build a :class:`RunState` snapshot of the current optimization state.

        This is the single place that converts the Objective's internal
        histories/counters into the canonical, serializer-agnostic
        :class:`RunState` data contract.
        """
        best_params = (
            np.asarray(self._best_params)
            if self._best_params is not None
            else np.array([])
        )
        return RunState(
            loss_history=np.asarray(self._loss_history, dtype=object),
            grad_history=np.asarray(self._grad_history, dtype=object),
            hessian_history=np.asarray(self._hessian_history, dtype=object),
            params_history=np.asarray(self._params_history, dtype=object),
            eval_type_history=np.asarray(self._eval_type_history, dtype=object),
            time_steps=np.asarray(self._time_steps, dtype=object),
            eval_count=self._eval_count,
            best_loss=float(self._best_loss),
            best_params=best_params,
            improvement_count=int(self._improvement_count),
            evals_since_improvement=int(self._evals_since_improvement),
            log_call_count=int(self._log_call_count),
            eval_type_counts=dict(self._eval_type_counts),
            metadata=self._build_metadata(algorithm_name),
        )

    def _apply_run_state(self, state: RunState) -> None:
        """Restore internal tracking state from a :class:`RunState`."""
        self._loss_history = (
            state.loss_history.tolist() if state.loss_history.size else []
        )
        self._grad_history = (
            state.grad_history.tolist() if state.grad_history.size else []
        )
        self._hessian_history = (
            state.hessian_history.tolist() if state.hessian_history.size else []
        )
        self._params_history = (
            state.params_history.tolist() if state.params_history.size else []
        )
        self._eval_type_history = (
            state.eval_type_history.tolist() if state.eval_type_history.size else []
        )
        self._time_steps = state.time_steps.tolist() if state.time_steps.size else []

        self._eval_count = int(state.eval_count)
        self._best_loss = jnp.array(state.best_loss)
        self._best_params = (
            jnp.array(state.best_params) if state.best_params.size > 0 else None
        )
        self._improvement_count = int(state.improvement_count)
        self._evals_since_improvement = int(state.evals_since_improvement)
        self._log_call_count = int(state.log_call_count)
        self._eval_type_counts = dict(state.eval_type_counts)

        # Store elapsed time as offset; leave _start_time as None so that
        # warmup_*() and start_logging() work correctly after loading.
        self._time_offset = float(self._time_steps[-1]) if self._time_steps else 0.0
        self._start_time = None

        # Update budget tracking
        if self._max_evals is not None:
            self._evals_left = max(0, self._max_evals - self._eval_count)
            self._evals_exceeded = self._evals_left <= 0

        # Adopt the loaded metadata's timestamp so subsequent saves/exports
        # use the original run's identity.
        if state.metadata.timestamp:
            self._timestamp = state.metadata.timestamp
        if state.metadata.algorithm_name:
            self.algorithm_str = state.metadata.algorithm_name

    def save_run_data(
        self,
        algorithm_name: str | None = None,
        filepath: str | None = None,
        hyper_param_str: str | None = None,
    ) -> Path:
        """Save current optimization state to a checkpoint file.

        Delegates to the configured :class:`CheckpointManager`, which
        serializes via :attr:`_serializer` and writes via
        :attr:`_backend` (atomic on the local filesystem). The path is
        resolved by :attr:`_resolver` unless ``filepath`` is given.

        Args:
            algorithm_name: Name of the algorithm for file naming.
                Defaults to ``self.algorithm_str`` if set, otherwise
                ``"unknown"``.
            filepath: Custom file path. If None, uses the resolver's
                structured naming convention.
            hyper_param_str: Optional hyperparameter string for
                subdirectory organization (e.g., "lr0.1_patience500").

        Returns:
            Path to the saved run data file.

        Example:
            >>> obj.save_run_data(algorithm_name="adam_gd")
            Path('data/objective_run_data/time100s_evals1000/voyager_adam_gd_2026-01-26_15-30-45.npz')
            >>> obj.save_run_data(algorithm_name="adam_gd", hyper_param_str="lr0.1")
            Path('data/objective_run_data/time100s_evals1000/lr0.1/voyager_adam_gd_2026-01-26_15-30-45.npz')
        """
        if algorithm_name is None:
            algorithm_name = self.algorithm_str or "unknown"

        # Refresh metadata with the caller's algorithm/hyperparam choices so
        # the serializer records them and the resolver paths correctly.
        state = self._build_run_state(algorithm_name)
        state.metadata.algorithm_name = algorithm_name
        if hyper_param_str is not None:
            state.metadata.hyper_param_str = hyper_param_str

        save_path = self._checkpoint_manager.save(
            state,
            explicit_path=filepath,
            hyper_param_str=hyper_param_str,
        )
        if self._verbose >= 1:
            print(f"Run data saved to {save_path}")
        return save_path

    def load_run_data(self, filepath: str | Path) -> None:
        """Load optimization state from a run data file.

        Restores all tracking state including loss history, parameters,
        and timing via the configured :class:`CheckpointManager`. The
        previously elapsed time is stored as an offset so that
        ``warmup_*()`` and ``start_logging()`` still work normally after
        loading.  Call ``start_logging()`` to resume the wall-clock timer.

        Args:
            filepath: Path to the run data checkpoint file to load.

        Raises:
            FileNotFoundError: If run data file doesn't exist.

        Example:
            >>> obj.load_run_data("data/objective_run_data/.../voyager_adam_gd_2026-01-26_15-30-45.npz")
            >>> obj.warmup_value_and_grad()   # OK — logging not yet active
            >>> obj.start_logging()           # resume wall-clock timer
            >>> print(f"Resuming from {obj.eval_count} evaluations")
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Run data file not found: {filepath}")

        state = self._checkpoint_manager.load(filepath)
        self._apply_run_state(state)

        # Warn if the checkpoint's save config differs from this Objective's
        loaded_cfg = state.metadata.extra.get("save_config")
        if loaded_cfg is not None:
            ckpt_cfg = SaveConfig.from_dict(loaded_cfg)
            diffs = self._save_config.mismatch(ckpt_cfg)
            if diffs and self._verbose >= 1:
                print(
                    "Warning: checkpoint save_config differs from current "
                    f"Objective in: {', '.join(diffs)}. Histories may be "
                    "inconsistent."
                )

        if self._verbose >= 1:
            print(f"Checkpoint loaded from {filepath}")
            if len(self._time_steps) > 0:
                print(
                    f"Resuming from: {self._eval_count} evals, {self._time_steps[-1]:.2f}s elapsed"
                )
            else:
                print(f"Resuming from: {self._eval_count} evals, 0.00s elapsed")

    def get_summary(self) -> dict:
        """Get a summary dictionary of the optimization run."""
        current_loss = self.current_loss
        if current_loss is not None:
            # Handle both scalar and batch losses
            ndim = self._ndim(current_loss)
            if ndim == 0:
                current_loss_value = float(current_loss)
            elif ndim == 1:
                # For batches, use min
                current_loss_value = float(jnp.nanmin(current_loss))
            else:
                current_loss_value = None
        else:
            current_loss_value = None

        return {
            "eval_count": self._eval_count,
            "time_elapsed": self.time_elapsed,
            "best_loss": float(self._best_loss) if self._best_loss != jnp.inf else None,
            "current_loss": current_loss_value,
            "improvement_count": self.improvement_count,
            "evals_since_improvement": self.evals_since_improvement,
            "budget_exceeded": self.budget_exceeded,
            "time_exceeded": self.time_exceeded,
            "evals_exceeded": self._evals_exceeded,
        }

    def reset(self) -> None:
        """Reset all tracking state for a new optimization run.

        Clears all histories, resets counters, and prepares for a fresh start.
        Does not modify the problem or budget limits.
        """
        self._start_time = None
        self._time_offset = 0.0
        self._eval_count = 0
        self._evals_left = self._max_evals
        self._evals_exceeded = False

        self._loss_history = []
        self._grad_history = []
        self._hessian_history = []
        self._params_history = []
        self._time_steps = []

        self._best_loss = jnp.inf
        self._best_params = None
        self._improvement_count = 0
        self._evals_since_improvement = 0
        self._eval_type_history = []
        self._log_call_count = 0
        self._eval_type_counts = {}
        self._display = None  # re-create on next render
        # Rebuild the storage stack so the next run starts with fresh path
        # caching / last-checkpoint state, preserving the format, directory,
        # and save_every cadence configured at construction.
        self._build_storage(
            self._checkpoint_format,
            self._checkpoint_dir,
            self._checkpoint_manager.save_every,
        )
        # New run -> new timestamp so saves do not silently overwrite the
        # previous run's checkpoint at the cached path.
        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def output_to_files(
        self,
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
    ) -> Path:
        """Output optimization results to human-readable files (plots + JSON).

        Delegates to the configured :class:`RunDataExporter`, which derives
        all artifacts from a :class:`RunState` snapshot of the current run
        plus the underlying problem (for sensitivity plots on optical
        problems). This keeps plotting and JSON writing out of the
        Objective and the checkpoint path.

        Files are saved under ``./data/problem_output/{problem_name}/
        {algorithm_str}/{hyper_param_str}/`` by default (configurable via
        the exporter's ``root``).

        Args:
            hyper_param_str: Hyperparameter string for directory naming
                (e.g., "lr0.1_patience500").
            hyper_param_str_in_filename: Whether to include hyperparams in
                filename.

        Returns:
            Path to the output directory.
        """
        state = self._build_run_state()
        return self._exporter.export(
            state,
            problem=self._problem,
            hyper_param_str=hyper_param_str,
            hyper_param_str_in_filename=hyper_param_str_in_filename,
        )
