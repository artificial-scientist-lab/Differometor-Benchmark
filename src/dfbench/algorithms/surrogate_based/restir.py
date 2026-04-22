"""ReSTIR: Resampled Surrogate-based Importance Sampling for Optimization.

Implementation of a kNN-surrogate based optimization algorithm using JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench import Objective


# --------- kNN Regressor ---------


def standardize_data(X_train, X_query=None):
    """Standardize data to zero mean and unit variance.

    Args:
        X_train: Training data (N, D) - used to compute mean/std
        X_query: Optional query data (M, D) - transformed using train statistics

    Returns:
        If X_query is None: (X_train_scaled, mean, std)
        If X_query provided: (X_train_scaled, X_query_scaled, mean, std)
    """
    mean = jnp.mean(X_train, axis=0, keepdims=True)
    std = jnp.std(X_train, axis=0, keepdims=True) + 1e-8  # Avoid division by zero

    X_train_scaled = (X_train - mean) / std

    if X_query is None:
        return X_train_scaled, mean, std

    X_query_scaled = (X_query - mean) / std
    return X_train_scaled, X_query_scaled, mean, std


@partial(jax.jit, static_argnames=("k",))
def knn_predict(X_train, y_train, X_query, k=10):
    """Batched kNN regression in pure JAX - runs entirely on GPU.

    Uses brute-force distance computation with inverse distance weighting.
    Optimized for cases where training set fits in memory.

    Args:
        X_train: Training points (N, D) - already standardized
        y_train: Training targets (N,)
        X_query: Query points (M, D) - already standardized
        k: Number of neighbors

    Returns:
        Predicted values (M,)
    """
    # Pairwise squared distances: (M, N) via broadcasting
    diffs = X_query[:, None, :] - X_train[None, :, :]  # (M, N, D)
    dists = jnp.sum(diffs**2, axis=-1)  # (M, N)

    # Top-k nearest neighbors (JAX uses partial sort - O(N) not O(N log N))
    neg_dists = -dists
    _, topk_idx = jax.lax.top_k(neg_dists, k)  # (M, k)
    topk_dists = jnp.take_along_axis(dists, topk_idx, axis=1)  # (M, k)

    # Distance-weighted average (inverse distance weighting)
    weights = 1.0 / (jnp.sqrt(topk_dists) + 1e-8)  # (M, k)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)

    topk_losses = y_train[topk_idx]  # (M, k)
    return jnp.sum(weights * topk_losses, axis=1)  # (M,)


def knn_predict_batched(X_train, y_train, X_query, k=10, batch_size=50_000):
    """Memory-safe batched kNN - stays in JAX, no copies.

    For large query sets (M > 100k), process in chunks to avoid OOM.

    Args:
        X_train: Training points (N, D) - already standardized
        y_train: Training targets (N,)
        X_query: Query points (M, D) - already standardized
        k: Number of neighbors
        batch_size: Number of queries to process at once

    Returns:
        Predicted values (M,)
    """
    n_queries = X_query.shape[0]
    preds = []

    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        batch = X_query[i:end]
        batch_pred = knn_predict(X_train, y_train, batch, k)
        preds.append(batch_pred)

    return jnp.concatenate(preds)


def importance(loss, tau=1.0):
    """Convert loss to importance weights using a softmax-like transformation."""
    return jnp.exp(-loss / tau)


# --------- ReSTIR Algorithm ---------


class ReSTIR(OptimizationAlgorithm):
    """ReSTIR: Resampled Spatio-Temporal Importance Sampling optimization.

    A surrogate-based optimization algorithm that uses k-nearest neighbors
    regression to estimate the loss surface and guide sampling towards
    promising regions. Uses pure JAX for GPU-accelerated kNN queries.

    Attributes:
        algorithm_str (str): Unique identifier for this algorithm.
        algorithm_type (AlgorithmType): Classification type.
        batch_size (int): Number of samples to evaluate per batch.
    """

    algorithm_str: str = "ReSTIR"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(
        self,
        batch_size: int = 1,
        knn_batch_size: int = 50_000,
    ) -> None:
        """Initialize the ReSTIR algorithm.

        Args:
            batch_size: Number of samples to evaluate per batch during optimization.
            knn_batch_size: Batch size for kNN queries (memory management).
        """
        self.batch_size = batch_size
        self.knn_batch_size = knn_batch_size

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        # ReSTIR-specific hyperparameters
        n_total_samples: int = 100_000,
        n_initial_reference_samples: int = 1000,
        reservoir_size: int = 1000,
        n_gd_candidates: int = 50,
        k_neighbors: int = 10,
        temperature: float = 1.0,
        gd_steps: int = 20,
        gd_learning_rate: float = 0.1,
    ) -> None:
        """Run the ReSTIR optimization algorithm.

        Uses k-nearest neighbors regression to estimate the loss surface for
        a large set of candidate samples, then selects the most promising ones
        for actual evaluation.

        Args:
            problem_objective: Pre-configured Objective instance. Use this for all
                function evaluations - it handles tracking automatically.
            max_iterations: Maximum number of algorithm iterations (not evaluations).
                If None, runs until budget is exceeded.
            init_params: Initial parameters (currently unused - generates initial samples).
            random_seed: Random seed for reproducibility.
            reservoir_size: Reservoir size for importance sampling.
            n_initial_reference_samples: Number of initial samples to evaluate (the real loss) for training kNN.
            n_total_samples: Number of candidate samples to generate per iteration.
            k_neighbors: Number of nearest neighbors for kNN regression.
            temperature: Temperature parameter for importance weighting. Higher means
                more exploration (flatter weights), lower means more exploitation.
            gd_steps: Number of Adam gradient steps to run for each selected GD candidate.
            gd_learning_rate: Learning rate for Optax Adam updates during GD refinement.
        """
        # 1. Setup
        obj = problem_objective
        problem = obj.problem

        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        # 3. JIT warmup (before start_logging so compilation time is excluded)
        obj.warmup_vmap_value(batch_size=self.batch_size)
        obj.warmup_vmap_grad(batch_size=self.batch_size)

        # 4. Start logging
        obj.start_logging()

        # 5. Generate initial reference set and evaluate
        knn_reference_points = obj.random_params_bounded(
            n_samples=n_initial_reference_samples
        )
        knn_reference_losses = jnp.empty(n_initial_reference_samples, dtype=jnp.float32)

        # Evaluate initial samples in batches
        for i in range(0, n_initial_reference_samples, self.batch_size):
            end = min(i + self.batch_size, n_initial_reference_samples)
            batch = knn_reference_points[i:end]
            knn_reference_losses = knn_reference_losses.at[i:end].set(
                obj.vmap_value(batch)
            )

        # 6. Main optimization loop
        iteration = 0
        while not obj.budget_exceeded:
            # Check max_iterations
            if max_iterations is not None and iteration >= max_iterations:
                break

            # Standardize reference data (fixed shape: n_initial_reference_samples)
            X_train_scaled, mean, std = standardize_data(knn_reference_points)
            y_train = knn_reference_losses

            # Always draw exactly n_total_samples candidates → fixed query shape → no JAX retrace
            drawn_samples = obj.random_params_bounded(n_samples=n_total_samples)
            X_query_scaled = (drawn_samples - mean) / std

            # Predict losses using kNN surrogate
            predicted_losses = knn_predict_batched(
                X_train_scaled,
                y_train,
                X_query_scaled,
                k=k_neighbors,
                batch_size=self.knn_batch_size,
            )

            # --- Reservoir sampling with importance weights ---
            # w(x) = hat(p)(x)/p(x) = importance (p(x) constant for uniform sampling)

            n_total = n_total_samples + n_initial_reference_samples

            # Concatenate predicted + reference, with origin-tracking mask
            all_losses = jnp.concatenate([predicted_losses, knn_reference_losses])
            all_samples = jnp.concatenate([drawn_samples, knn_reference_points])
            is_predicted = jnp.arange(n_total) < n_total_samples

            # Shuffle all arrays together via shared permutation
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n_total)
            all_losses, all_samples, is_predicted = (
                all_losses[perm],
                all_samples[perm],
                is_predicted[perm],
            )

            # Pad + reshape into reservoirs of size reservoir_size
            n_reservoirs = -(-n_total // reservoir_size)  # ceil division
            n_pad = n_reservoirs * reservoir_size - n_total
            r_losses = jnp.pad(all_losses, (0, n_pad)).reshape(
                n_reservoirs, reservoir_size
            )
            r_samples = jnp.pad(all_samples, ((0, n_pad), (0, 0))).reshape(
                n_reservoirs, reservoir_size, -1
            )
            r_is_pred = jnp.pad(is_predicted, (0, n_pad)).reshape(
                n_reservoirs, reservoir_size
            )

            # Valid-entry mask (last reservoir may be partial)
            sizes = (
                jnp.full(n_reservoirs, reservoir_size)
                .at[-1]
                .set(reservoir_size - n_pad)
            )  # Always reservoir_size except last one
            valid_mask = (
                jnp.arange(reservoir_size)[None, :] < sizes[:, None]
            )  # (n_reservoirs, reservoir_size)

            # Importance-weighted categorical sampling per reservoir
            r_logits = jnp.where(valid_mask, -r_losses / temperature, -jnp.inf)

            keys = jax.random.split(key, n_reservoirs + 1)
            key, subkeys = keys[0], keys[1:]

            # Select one sample per reservoir according to importance weights
            r_selected_idx = jax.vmap(
                lambda k, logits: jax.random.categorical(k, logits, shape=())
            )(subkeys, r_logits)  # (n_reservoirs,)

            # Gather selected entries per reservoir
            reservoir_indices = jnp.arange(n_reservoirs)
            sel_samples = r_samples[
                reservoir_indices, r_selected_idx
            ]  # (n_reservoirs, D)
            sel_losses = r_losses[reservoir_indices, r_selected_idx]  # (n_reservoirs,)
            sel_is_pred = r_is_pred[
                reservoir_indices, r_selected_idx
            ]  # (n_reservoirs,)

            # --- Evaluate true losses (only surrogate-predicted need obj calls) ---
            n_pred = int(jnp.sum(sel_is_pred).item())
            # Reference samples keep their known losses; predicted start at inf
            true_losses = jnp.where(sel_is_pred, jnp.inf, sel_losses)

            if n_pred > 0:
                # All predicted samples to evaluate
                pred_samples = sel_samples[sel_is_pred]
                # Evaluate in batches to respect batch_size
                evaluated = jnp.concatenate(
                    [
                        obj.vmap_value(
                            pred_samples[i : min(i + self.batch_size, n_pred)]
                        )
                        for i in range(0, n_pred, self.batch_size)
                    ]
                )
                # Replace all predicted losses with true evaluated values
                true_losses = true_losses.at[sel_is_pred].set(evaluated)

            # Resampling weights: correct for surrogate prediction error
            # (zero for reference samples since sel_losses == true_losses)
            # weight_logits = (sel_losses - true_losses) / temperature
            # TODO think about this, I don't think it would be fair to weight estimated
            # Samples by their "surprise" factor sel_losses - true_losses, because the
            # reference losses will always have zero surprise independently of how good
            # they are...
            weight_logits = -true_losses / temperature

            # Select K final samples via Gumbel top-K (categorical without replacement)
            key, subkey = jax.random.split(key)
            n_final_candidates = min(n_gd_candidates, n_reservoirs)
            final_idx = jax.random.categorical(
                subkey,
                weight_logits,
                shape=(n_final_candidates,),
                replace=False,
            )
            selected_samples = sel_samples[final_idx]
            selected_logits = weight_logits[final_idx]

            # Batched Adam GD refinement on all selected candidates
            if n_final_candidates > 0 and gd_steps > 0:
                lower, upper = problem.bounds
                gd_samples = selected_samples
                gd_optimizer = optax.adam(gd_learning_rate)
                gd_opt_state = gd_optimizer.init(gd_samples)

                for _ in range(gd_steps):
                    if obj.budget_exceeded:
                        break
                    gd_grads = obj.vmap_grad(gd_samples)
                    gd_updates, gd_opt_state = gd_optimizer.update(
                        gd_grads, gd_opt_state, gd_samples
                    )
                    gd_samples = optax.apply_updates(gd_samples, gd_updates)
                    gd_samples = jnp.clip(gd_samples, lower, upper)

                n_gd = int(gd_samples.shape[0])
                gd_losses = jnp.concatenate(
                    [
                        obj.vmap_value(gd_samples[i : min(i + self.batch_size, n_gd)])
                        for i in range(0, n_gd, self.batch_size)
                    ]
                )

                # Update reference set with refined points, keeping it at fixed size
                # (n_initial_reference_samples) so kNN training shape never changes.
                all_ref_points = jnp.concatenate(
                    [knn_reference_points, gd_samples], axis=0
                )
                all_ref_losses = jnp.concatenate(
                    [knn_reference_losses, gd_losses], axis=0
                )
                keep_idx = jnp.argsort(all_ref_losses)[:n_initial_reference_samples]
                knn_reference_points = all_ref_points[keep_idx]
                knn_reference_losses = all_ref_losses[keep_idx]

            iteration += 1
