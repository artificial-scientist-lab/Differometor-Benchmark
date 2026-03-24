"""TuRBO→L-BFGS — Trust-region BO followed by local gradient refinement.

Runs a TuRBO phase (local trust-region Bayesian Optimization) to locate a
promising basin, then hands off the best incumbent to the Optax L-BFGS
local-refinement phase for rapid gradient-based convergence.

This combines the global exploration strength of TuRBO with the fast local
convergence of L-BFGS, making it particularly suitable for problems that are
expensive to evaluate globally but have smooth gradients near the optimum.

Operates in **bounded** parameter space throughout. The TuRBO phase evaluates
the objective directly; the L-BFGS phase runs on the sigmoid objective
internally and logs bounded params via ``log_evaluation``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.core.utils import t2j, inverse_sigmoid_bounding
from dfbench.algorithms.surrogate_based.botorch_turbo import (
    BotorchTuRBO,
    TurboState,
    update_turbo_state,
)
from dfbench.algorithms.surrogate_based._botorch_common import (
    DEVICE,
    DTYPE,
    evaluate_objective,
    fit_gp,
    get_problem_bounds_torch,
    sobol_initial_samples,
)


class TuRBOLBFGS(OptimizationAlgorithm):
    """TuRBO→L-BFGS: trust-region BO + local gradient refinement.

    Phase 1 — **TuRBO**: runs local trust-region BO in bounded parameter space
    to find a promising region.

    Phase 2 — **L-BFGS** (via Optax): takes the best incumbent from Phase 1,
    maps it to unbounded (sigmoid) space internally, and runs L-BFGS for rapid
    local convergence. Results are logged back via ``log_evaluation`` using
    bounded params so the Objective stays in bounded mode throughout.

    The TuRBO phase uses the existing ``BotorchTuRBO`` infrastructure.
    The L-BFGS phase mirrors the ``LBFGSGD`` implementation.

    Attributes:
        algorithm_str: ``"turbo_lbfgs"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "turbo_lbfgs"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        self.device = DEVICE
        self.dtype = DTYPE

    @staticmethod
    def _linesearch_eval_count(opt_state) -> int:
        """Extract the number of internal line-search evaluations from Optax LBFGS."""
        states = opt_state if isinstance(opt_state, tuple) else (opt_state,)
        for state in reversed(states):
            info = getattr(state, "info", None)
            if info is None:
                continue
            num_steps = getattr(info, "num_linesearch_steps", None)
            if num_steps is not None:
                return int(jax.device_get(num_steps))
        return 0

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        # TuRBO phase
        turbo_iterations: int | None = None,
        n_initial: int | None = None,
        turbo_batch_size: int = 1,
        turbo_acqf: str = "ts",
        # L-BFGS phase
        lbfgs_patience: int = 200,
        **kwargs,
    ) -> None:
        """Run TuRBO→L-BFGS.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            turbo_iterations: Max BO iterations for the TuRBO phase. Required.
            n_initial: Sobol initialisation for TuRBO. Defaults to ``2 * dim``.
            turbo_batch_size: Candidates per TuRBO iteration.
            turbo_acqf: Acquisition function: ``"ts"`` or ``"ei"``.
            lbfgs_patience: Stop L-BFGS after this many iterations without
                improvement.
            **kwargs: Additional TuRBO kwargs.
        """
        if turbo_iterations is None:
            raise ValueError("turbo_iterations is required")

        obj = problem_objective
        problem = obj.problem
        dim = problem.n_params

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        if n_initial is None:
            n_initial = 2 * dim

        # ════════════════════════════════════════════════════════════════
        # Phase 1: TuRBO
        # ════════════════════════════════════════════════════════════════

        bounds_torch = get_problem_bounds_torch(problem, self.device, self.dtype)

        turbo_engine = BotorchTuRBO()
        turbo_engine.device = self.device
        turbo_engine.dtype = self.dtype

        # JIT warmup (bounded)
        _ = obj.vmap_value(jnp.zeros((1, dim)))

        obj.start_logging()

        # ── Run a single TuRBO instance ──────────────────────────────
        sobol = torch.quasirandom.SobolEngine(dim, scramble=True, seed=random_seed)
        train_X = sobol.draw(n_initial).to(device=self.device, dtype=self.dtype)

        if init_params is not None:
            from botorch.utils.transforms import normalize
            x0 = torch.tensor(
                np.asarray(init_params).reshape(1, -1), device=self.device, dtype=self.dtype
            )
            train_X = torch.cat([normalize(x0, bounds_torch), train_X])

        Y_raw, valid = evaluate_objective(train_X, bounds_torch, obj)
        train_X = train_X[valid]
        train_Y = Y_raw[valid].unsqueeze(-1)

        if len(train_Y) == 0:
            raise ValueError("All initial TuRBO evaluations returned NaN/Inf.")

        state = TurboState(
            dim=dim,
            batch_size=turbo_batch_size,
            best_value=train_Y.max().item(),
        )

        turbo_iter = 0
        while (
            not obj.budget_exceeded
            and not state.restart_triggered
            and turbo_iter < turbo_iterations
        ):
            Y_mean = train_Y.mean()
            Y_std = train_Y.std().clamp(min=1e-6)
            train_Y_norm = (train_Y - Y_mean) / Y_std

            model = fit_gp(train_X, train_Y_norm, turbo_constraints=True)
            model.eval()

            X_next = turbo_engine._generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y_norm,
                batch_size=turbo_batch_size,
                acqf=turbo_acqf,
            )

            Y_next, vm = evaluate_objective(X_next, bounds_torch, obj)
            Y_next = Y_next.unsqueeze(-1)

            if vm.any():
                state = update_turbo_state(state, Y_next[vm])
                train_X = torch.cat([train_X, X_next[vm]])
                train_Y = torch.cat([train_Y, Y_next[vm]])
            else:
                state.failure_counter += 1
                if state.failure_counter >= state.failure_tolerance:
                    state.length /= 2.0
                    state.failure_counter = 0
                if state.length < state.length_min:
                    state.restart_triggered = True

            turbo_iter += 1

        if obj.budget_exceeded:
            return

        # ════════════════════════════════════════════════════════════════
        # Phase 2: L-BFGS refinement
        # ════════════════════════════════════════════════════════════════
        # The Objective stays in bounded mode (cannot switch after logging
        # starts).  We run L-BFGS on the sigmoid objective directly and
        # log evaluations manually via log_evaluation with the bounded
        # params and the loss.

        from differometor.utils import sigmoid_bounding

        best_bounded = jnp.asarray(obj.best_params_bounded)

        # Map best bounded → unbounded
        bounds_jax = jnp.asarray(problem.bounds)
        params = inverse_sigmoid_bounding(best_bounded, bounds_jax)

        # Build JIT-compiled L-BFGS step using the sigmoid objective,
        # which lives entirely in unbounded space.
        value_fn = problem.sigmoid_objective_function
        value_and_grad_fn = jax.value_and_grad(value_fn)

        def _to_bounded(unbounded_params):
            return sigmoid_bounding(unbounded_params, bounds_jax)

        optimizer = optax.lbfgs()
        optimizer_state = optimizer.init(params)

        @jax.jit
        def _step(params, opt_state):
            loss, grads = value_and_grad_fn(params)
            updates, new_state = optimizer.update(
                grads, opt_state, params,
                value=loss, grad=grads, value_fn=value_fn,
            )
            new_params = optax.apply_updates(params, updates)
            return jnp.asarray(new_params), new_state, loss, grads

        # L-BFGS warmup (use random bounded, map to unbounded)
        rng_params = obj.random_params_bounded()
        warmup_unb = inverse_sigmoid_bounding(rng_params, bounds_jax)
        warmup_state = optimizer.init(warmup_unb)
        rng_params2 = obj.random_params_bounded()
        warmup_unb2 = inverse_sigmoid_bounding(rng_params2, bounds_jax)
        _, warmup_state, _, _ = _step(warmup_unb2, warmup_state)
        rng_params3 = obj.random_params_bounded()
        warmup_unb3 = inverse_sigmoid_bounding(rng_params3, bounds_jax)
        _ = _step(warmup_unb3, warmup_state)

        while not obj.budget_exceeded:
            prior_params = params
            params, optimizer_state, loss, grads = _step(params, optimizer_state)

            # Log using the bounded params so the Objective stays consistent
            bounded_params = _to_bounded(prior_params)
            obj.log_evaluation(bounded_params, loss)

            extra = self._linesearch_eval_count(optimizer_state)
            for _ in range(extra):
                if obj.budget_exceeded:
                    break
                obj.log_evaluation(bounded_params, loss)

            if lbfgs_patience is not None and obj.evals_since_improvement > lbfgs_patience:
                break
