"""Shared SciPy minimize integration for gradient-based algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.optimize import BFGS as ScipyBFGS
from scipy.optimize import Bounds
from scipy.optimize import minimize

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


class SciPyBudgetExceeded(RuntimeError):
    """Raised internally to stop SciPy once the Objective budget is exhausted."""


@dataclass(slots=True)
class SciPyConfig:
    """Per-method SciPy integration settings."""

    method: str
    unbounded: bool
    use_bounds: bool
    use_jac: bool = True
    use_hessp: bool = False
    use_dense_hessian: bool = False
    hessian_update_strategy: object | None = None
    cache_hessp: bool = False


class SciPyObjectiveAdapter:
    """Adapt ``Objective`` to SciPy call signatures with fair logging."""

    def __init__(self, obj: Objective, config: SciPyConfig) -> None:
        self.obj = obj
        self.config = config

        func = obj._func
        grad_fn = jax.grad(func)

        self._value_fn = jax.jit(func)
        self._value_and_grad_fn = None
        self._hessp_only_fn = None
        self._value_grad_and_hessp_fn = None
        self._value_grad_and_hessian_fn = None

        if config.use_jac or config.use_hessp or config.use_dense_hessian:
            self._value_and_grad_fn = jax.jit(jax.value_and_grad(func))

        if config.use_hessp:

            def _value_grad_and_hessp(params, vector):
                value, grad = jax.value_and_grad(func)(params)
                hessp = jax.jvp(grad_fn, (params,), (vector,))[1]
                return value, grad, hessp

            def _hessp_only(params, vector):
                return jax.jvp(grad_fn, (params,), (vector,))[1]

            self._value_grad_and_hessp_fn = jax.jit(_value_grad_and_hessp)
            self._hessp_only_fn = jax.jit(_hessp_only)

        if config.use_dense_hessian:

            def _value_grad_and_hessian(params):
                value, grad = jax.value_and_grad(func)(params)
                hessian = jax.hessian(func)(params)
                return value, grad, hessian

            self._value_grad_and_hessian_fn = jax.jit(_value_grad_and_hessian)

        self._latest_x: np.ndarray | None = None
        self._latest_loss = None
        self._latest_grad = None
        self._latest_hessian = None

        self._latest_hessp_key: tuple[bytes, bytes] | None = None
        self._latest_hessp = None

    @property
    def bounds(self) -> Bounds | None:
        """Return SciPy bounds in physical space when requested."""
        if not self.config.use_bounds:
            return None
        problem_bounds = np.asarray(self.obj.problem.bounds, dtype=float)
        return Bounds(problem_bounds[0], problem_bounds[1], keep_feasible=True)

    @property
    def constraints(self) -> tuple:
        """Return supported SciPy constraints or fail loudly."""
        problem = self.obj.problem
        for attr_name in (
            "scipy_constraints",
            "constraints",
            "linear_constraints",
            "nonlinear_constraints",
        ):
            if not hasattr(problem, attr_name):
                continue
            value = getattr(problem, attr_name)
            if value not in (None, (), [], {}):
                raise NotImplementedError(
                    "Problem exposes constraint metadata via "
                    f"'{attr_name}', but this batch only supports box constraints."
                )
        return ()

    def warmup(self) -> None:
        """Warm up the exact JAX paths this SciPy adapter will use."""
        params = jnp.asarray(self.obj._deterministic_warmup_params())
        self._warmup_twice(self._value_fn, params)

        if self._value_and_grad_fn is not None:
            self._warmup_twice(self._value_and_grad_fn, params)

        if self._value_grad_and_hessp_fn is not None:
            vector = jnp.ones_like(params)
            self._warmup_twice(self._value_grad_and_hessp_fn, params, vector)
            self._warmup_twice(self._hessp_only_fn, params, vector)

        if self._value_grad_and_hessian_fn is not None:
            self._warmup_twice(self._value_grad_and_hessian_fn, params)

    def _warmup_twice(self, fn, *args) -> None:
        fn(*args)
        fn(*args)

    def _to_numpy_vector(self, x: Float[Array, "..."] | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return np.atleast_1d(arr).astype(float, copy=False)

    def _same_point(self, x: np.ndarray) -> bool:
        return self._latest_x is not None and np.array_equal(self._latest_x, x)

    def _check_budget(self) -> None:
        if self.obj.budget_exceeded:
            raise SciPyBudgetExceeded("Objective budget exhausted inside SciPy.")

    # Penalty returned to SciPy when the objective produces non-finite values.
    _NAN_PENALTY: float = 1e10

    def _cache_value_grad(
        self,
        x: np.ndarray,
        loss,
        grad,
        hessian=None,
    ) -> None:
        self._latest_x = np.array(x, copy=True)
        self._latest_loss = loss
        self._latest_grad = grad
        self._latest_hessian = hessian
        self._latest_hessp_key = None
        self._latest_hessp = None

    @staticmethod
    def _sanitize_loss(loss: float) -> float:
        """Replace non-finite loss with a large penalty so SciPy retreats."""
        if not np.isfinite(loss):
            return SciPyObjectiveAdapter._NAN_PENALTY
        return loss

    @staticmethod
    def _sanitize_grad(grad: np.ndarray) -> np.ndarray:
        """Replace non-finite gradient entries with zeros."""
        if not np.all(np.isfinite(grad)):
            return np.where(np.isfinite(grad), grad, 0.0)
        return grad

    def evaluate_value(self, x: np.ndarray) -> float:
        """Return loss only and log exactly once for this point."""
        x_np = self._to_numpy_vector(x)
        if self._same_point(x_np):
            return self._sanitize_loss(float(self._latest_loss))

        loss = self._value_fn(jnp.asarray(x_np))
        self.obj.log_evaluation(jnp.asarray(x_np), loss, None)
        self._cache_value_grad(x_np, loss, None)
        self._check_budget()
        return self._sanitize_loss(float(loss))

    def evaluate_value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Return loss/grad and log exactly once for this point."""
        if self._value_and_grad_fn is None:
            raise RuntimeError(
                "evaluate_value_and_grad() requested without jac-enabled adapter."
            )
        x_np = self._to_numpy_vector(x)
        if self._same_point(x_np):
            return (
                self._sanitize_loss(float(self._latest_loss)),
                self._sanitize_grad(np.asarray(self._latest_grad, dtype=float)),
            )

        loss, grad = self._value_and_grad_fn(jnp.asarray(x_np))
        self.obj.log_evaluation(jnp.asarray(x_np), loss, grad)
        self._cache_value_grad(x_np, loss, grad)
        self._check_budget()
        return (
            self._sanitize_loss(float(loss)),
            self._sanitize_grad(np.asarray(grad, dtype=float)),
        )

    def fun(self, x: np.ndarray) -> float:
        """SciPy ``fun`` callback."""
        if self.config.use_jac:
            loss, _ = self.evaluate_value_and_grad(x)
            return loss
        return self.evaluate_value(x)

    def jac(self, x: np.ndarray) -> np.ndarray:
        """SciPy ``jac`` callback."""
        _, grad = self.evaluate_value_and_grad(x)
        return grad

    def hessp(self, x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """SciPy ``hessp`` callback with explicit logging."""
        if self._value_grad_and_hessp_fn is None or self._hessp_only_fn is None:
            raise RuntimeError(
                "hessp() requested, but this adapter was not configured."
            )

        x_np = self._to_numpy_vector(x)
        vector_np = self._to_numpy_vector(vector)
        key = (x_np.tobytes(), vector_np.tobytes())
        if self.config.cache_hessp and self._latest_hessp_key == key:
            return np.asarray(self._latest_hessp, dtype=float)

        if self._same_point(x_np):
            hessp = self._hessp_only_fn(jnp.asarray(x_np), jnp.asarray(vector_np))
            self.obj.log_evaluation(
                jnp.asarray(x_np), self._latest_loss, self._latest_grad
            )
        else:
            loss, grad, hessp = self._value_grad_and_hessp_fn(
                jnp.asarray(x_np),
                jnp.asarray(vector_np),
            )
            self.obj.log_evaluation(jnp.asarray(x_np), loss, grad)
            self._cache_value_grad(x_np, loss, grad)

        self._latest_hessp_key = key
        self._latest_hessp = hessp
        self._check_budget()
        return np.asarray(hessp, dtype=float)

    def hess(self, x: np.ndarray) -> np.ndarray:
        """SciPy dense Hessian callback with explicit logging."""
        if self._value_grad_and_hessian_fn is None:
            raise RuntimeError("hess() requested, but this adapter was not configured.")

        x_np = self._to_numpy_vector(x)
        if self._same_point(x_np) and self._latest_hessian is not None:
            return np.asarray(self._latest_hessian, dtype=float)

        value, grad, hessian = self._value_grad_and_hessian_fn(jnp.asarray(x_np))
        hessian_np = np.asarray(hessian, dtype=float)
        if not np.all(np.isfinite(hessian_np)):
            raise ValueError("Dense Hessian contains non-finite values.")

        self.obj.log_evaluation(jnp.asarray(x_np), value, grad, hessian)
        self._cache_value_grad(x_np, value, grad, hessian=hessian)
        self._check_budget()
        return hessian_np


class ScipyMinimizeAlgorithm(OptimizationAlgorithm):
    """Shared implementation for SciPy ``minimize``-based optimizers."""

    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED
    scipy_config: SciPyConfig

    def __init__(self) -> None:
        self._last_result = None
        self._last_hessian_update_strategy = None

    def _resolve_init_params(
        self,
        obj: Objective,
        init_params: Float[Array, "..."] | None,
    ) -> Float[Array, "n_params"]:
        if init_params is not None:
            return jnp.asarray(init_params)
        if self.scipy_config.unbounded:
            return obj.random_params_unbounded()
        return obj.random_params_bounded()

    def _perturbed_best_params(
        self,
        obj: Objective,
        scale: float = 0.1,
    ) -> Float[Array, "n_params"] | None:
        """Return the best-known point with Gaussian perturbation.

        Returns None if no best params are available yet.
        """
        best = obj.best_params
        if best is None:
            return None
        best_np = np.asarray(best, dtype=float)
        noise = scale * np.random.randn(*best_np.shape)
        perturbed = best_np + noise
        if not self.scipy_config.unbounded and obj.bounds is not None:
            lb = np.asarray(obj.bounds[0], dtype=float)
            ub = np.asarray(obj.bounds[1], dtype=float)
            perturbed = np.clip(perturbed, lb, ub)
        return jnp.asarray(perturbed)

    def _run_scipy_minimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None,
        random_seed: int | None,
        tol: float | None,
        options: dict[str, object],
        *,
        hessian_update_strategy: object | None = None,
    ) -> None:
        obj = problem_objective
        self.prepare(
            obj, unbounded=self.scipy_config.unbounded, random_seed=random_seed
        )

        x0 = np.asarray(self._resolve_init_params(obj, init_params), dtype=float)
        adapter = SciPyObjectiveAdapter(
            obj,
            SciPyConfig(
                method=self.scipy_config.method,
                unbounded=self.scipy_config.unbounded,
                use_bounds=self.scipy_config.use_bounds,
                use_jac=self.scipy_config.use_jac,
                use_hessp=self.scipy_config.use_hessp,
                use_dense_hessian=self.scipy_config.use_dense_hessian,
                hessian_update_strategy=(
                    hessian_update_strategy
                    if hessian_update_strategy is not None
                    else self.scipy_config.hessian_update_strategy
                ),
                cache_hessp=self.scipy_config.cache_hessp,
            ),
        )
        self._last_hessian_update_strategy = adapter.config.hessian_update_strategy
        adapter.warmup()

        minimize_kwargs = {
            "fun": adapter.fun,
            "x0": x0,
            "method": adapter.config.method,
            "jac": adapter.jac if adapter.config.use_jac else None,
            "bounds": adapter.bounds,
            "constraints": adapter.constraints,
            "tol": tol,
            "options": {k: v for k, v in options.items() if v is not None},
        }

        if adapter.config.use_hessp:
            minimize_kwargs["hessp"] = adapter.hessp
        if adapter.config.use_dense_hessian:
            minimize_kwargs["hess"] = adapter.hess
        elif adapter.config.hessian_update_strategy is not None:
            minimize_kwargs["hess"] = adapter.config.hessian_update_strategy

        obj.start_logging()

        restart_count = 0
        while not obj.budget_exceeded:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self._last_result = minimize(**minimize_kwargs)
            except SciPyBudgetExceeded:
                self._last_result = None
                return

            # Budget not yet exhausted — restart with alternating strategy:
            #   even restarts → perturb best-known point (exploit basin)
            #   odd restarts  → fresh random point (explore new basin)
            if not obj.budget_exceeded:
                restart_count += 1
                if restart_count % 2 == 0:
                    x0_new = self._perturbed_best_params(obj)
                    if x0_new is None:
                        x0_new = self._resolve_init_params(obj, None)
                else:
                    x0_new = self._resolve_init_params(obj, None)
                minimize_kwargs["x0"] = np.asarray(x0_new, dtype=float)


def bfgs_hessian_update_strategy(**kwargs) -> ScipyBFGS:
    """Return SciPy's BFGS Hessian update strategy."""
    return ScipyBFGS(**kwargs)
