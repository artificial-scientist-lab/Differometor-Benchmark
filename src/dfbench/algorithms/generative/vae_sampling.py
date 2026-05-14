"""VAE-based optimization with Bayesian Optimization in learned latent space.

This module implements a two-phase optimization approach:
1. VAE Training Phase: Samples candidates from the active Objective space,
    evaluates them, and trains a Variational Autoencoder on high-quality samples
    to learn a compressed latent representation
2. BO Phase: Performs Bayesian Optimization in the learned latent space using
   Gaussian Process surrogate models and batch Expected Improvement acquisition

The VAE learns a low-dimensional (d/10) latent space that captures the structure
of high-quality solutions, making optimization more efficient in high dimensions.
"""

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Array, Float
from typing import Callable
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.optim import optimize_acqf
from botorch.generation import gen_candidates_scipy
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from dfbench.core.utils import t2j, j2t
from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class ResidualBlock(nn.Module):
    """Residual block with skip connections for VAE."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetVAE(nn.Module):
    """ResNet-style Variational Autoencoder with residual blocks."""

    def __init__(
        self,
        input_dim: int = 250,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
    ) -> None:
        super(ResNetVAE, self).__init__()

        # Encoder
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Mish())
        self.encoder_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Mish())
        self.decoder_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.encoder_blocks:
            h = block(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        for block in self.decoder_blocks:
            h = block(h)
        return self.final_layer(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss with reconstruction and KL divergence components."""
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + (beta * kld), recon_loss, kld


def train_vae(
    model: ResNetVAE,
    dataloader: DataLoader,
    epochs: int = 100,
    device: str = "cpu",
    verbose: bool = False,
    stop_training: Callable[[], bool] | None = None,
) -> None:
    """Train the VAE with cyclic KL annealing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        if stop_training is not None and stop_training():
            break

        total_loss = 0
        cycle_len = 20
        beta = min(1.0, (epoch % cycle_len) / (cycle_len * 0.5))

        for batch in dataloader:
            if stop_training is not None and stop_training():
                break

            x = batch[0].to(device)
            if x.shape[0] < 2:
                continue

            optimizer.zero_grad()

            recon_x, mu, logvar = model(x)
            loss, _, _ = vae_loss_function(recon_x, x, mu, logvar, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if verbose and epoch % 5 == 0:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"VAE Epoch {epoch} | Beta: {beta:.2f} | Loss: {avg_loss:.1f}")


class VAESampling(OptimizationAlgorithm):
    """VAE-based optimization with Bayesian Optimization in learned latent space.

        Implements a two-phase optimization approach:
        1. VAE Training Phase: Samples candidates through ``Objective.random_params()``,
             evaluates them, and trains a Variational Autoencoder on high-quality samples
             to learn a compressed latent representation
    2. BO Phase: Performs Bayesian Optimization in the learned latent space using
       Gaussian Process surrogate models and batch Expected Improvement acquisition

    The VAE learns a low-dimensional (d/10) latent space that captures the structure
    of high-quality solutions, making optimization more efficient in high dimensions.

    All history tracking is handled by the `Objective` wrapper.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("vae_sampling").
        algorithm_type (AlgorithmType): Type classification (GENERATIVE).

    Note:
        This algorithm uses `problem.sigmoid_objective_function` which applies
        sigmoid bounding to handle infinite parameter spaces during VAE training.
    """

    algorithm_str: str = "vae_sampling"
    algorithm_type: AlgorithmType = AlgorithmType.GENERATIVE

    def __init__(self, batch_size: int = 1) -> None:
        """Initialize VAESampling optimizer.

        Args:
            batch_size: Number of candidates evaluated per ``vmap_value`` call.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def _evaluate_candidates(
        self, obj: Objective, candidates: Float[Array, "n d"]
    ) -> Float[Array, "n"]:
        """Evaluate candidates through ``vmap_value`` using constructor batch size."""
        losses = []
        for start in range(0, candidates.shape[0], self.batch_size):
            losses.append(obj.vmap_value(candidates[start : start + self.batch_size]))
        return jnp.concatenate(losses)

    def _sample_training_data(
        self,
        obj: Objective,
        max_samples: int | None,
        sampling_budget_fraction: float,
    ) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
        """Sample and evaluate VAE training candidates within the budget split."""
        if max_samples is None and obj.evals_left is None and obj.time_left is None:
            raise ValueError(
                "vae_training_samples=None requires an Objective eval or time budget."
            )

        candidates_list = []
        losses_list = []
        samples_collected = 0

        while not obj.budget_exceeded:
            if obj.budget_progress_fraction >= sampling_budget_fraction:
                break
            if max_samples is not None and samples_collected >= max_samples:
                break

            chunk_size = self.batch_size
            if obj._max_evals is not None:
                evals_for_sampling = int(
                    np.floor(obj._max_evals * sampling_budget_fraction)
                )
                evals_left_for_sampling = evals_for_sampling - obj.eval_count
                chunk_size = min(chunk_size, evals_left_for_sampling)
            if max_samples is not None:
                chunk_size = min(chunk_size, max_samples - samples_collected)
            if obj.evals_left is not None:
                chunk_size = min(chunk_size, obj.evals_left)
            if chunk_size < 1:
                break

            candidates = obj.random_params(n_samples=chunk_size)
            candidates = jnp.atleast_2d(candidates)
            losses = obj.vmap_value(candidates)

            candidates_list.append(candidates)
            losses_list.append(losses)
            samples_collected += chunk_size

        if not candidates_list:
            raise ValueError(
                "No VAE training samples were evaluated before budget exhaustion."
            )

        return jnp.concatenate(candidates_list), jnp.concatenate(losses_list)

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        vae_training_samples: int | None = 1000,
        sampling_budget_fraction: float = 0.25,
        vae_epochs: int = 100,
        latent_dim_factor: int = 10,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        vae_train_batch_size: int = 32,
        top_k: float = 0.02,
        n_initial: int = 20,
        bo_batch_size: int = 16,
        acqf_raw_samples: int = 512,
        acqf_num_restarts: int = 4,
    ) -> None:
        """Run VAE training followed by Bayesian Optimization in latent space.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of BO iterations in latent space (required).
            init_params: Initial parameters to seed optimization (unused).
            random_seed: Random seed for reproducibility.
            vae_training_samples: Optional cap on samples to evaluate for VAE
                training. If None, sampling is controlled only by budget.
            sampling_budget_fraction: Stop the sampling phase once this fraction
                of the tightest Objective budget is consumed.
            vae_epochs: Number of epochs to train VAE.
            latent_dim_factor: Factor to determine latent dimension (d // latent_dim_factor + 1).
            hidden_dim: Hidden dimension for VAE architecture.
            num_blocks: Number of residual blocks in VAE.
            vae_train_batch_size: Mini-batch size for VAE training.
            top_k: Fraction of top-performing samples to select for VAE training
                after the objective-guided sampling phase.
            n_initial: Number of initial Sobol samples for BO phase.
            bo_batch_size: Number of latent BO candidates proposed per GP fit.
            acqf_raw_samples: Number of raw samples for acquisition optimization.
            acqf_num_restarts: Number of restarts for acquisition optimization.
        """
        if vae_train_batch_size < 1:
            raise ValueError("vae_train_batch_size must be at least 1.")
        if vae_training_samples is not None and vae_training_samples < 1:
            raise ValueError("vae_training_samples must be None or at least 1.")
        if not 0.0 < sampling_budget_fraction < 1.0:
            raise ValueError("sampling_budget_fraction must be between 0 and 1.")
        if not 0.0 < top_k <= 1.0:
            raise ValueError("top_k must be a fraction in (0, 1].")
        if bo_batch_size < 1:
            raise ValueError("bo_batch_size must be at least 1.")

        obj = problem_objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)
        torch.manual_seed(random_seed)

        input_dim = problem.n_params
        latent_dim = input_dim // latent_dim_factor + 1

        # Create VAE
        vae = ResNetVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_blocks,
        )
        vae.to(self._device)

        obj.warmup_vmap_value(batch_size=self.batch_size)

        obj.start_logging()

        # === VAE Training Phase ===
        candidates_jax, eval_losses = self._sample_training_data(
            obj=obj,
            max_samples=vae_training_samples,
            sampling_budget_fraction=sampling_budget_fraction,
        )

        top_count = max(2, int(np.ceil(candidates_jax.shape[0] * top_k)))
        top_count = min(top_count, candidates_jax.shape[0])
        top_indices = jnp.argsort(eval_losses)[:top_count]
        best_samples_jax = candidates_jax[top_indices]
        best_samples_torch = j2t(best_samples_jax).float().to(self._device)

        dataset = TensorDataset(best_samples_torch)
        loader = DataLoader(dataset, batch_size=vae_train_batch_size, shuffle=True)

        vae.train()
        train_vae(
            vae,
            loader,
            epochs=vae_epochs,
            device=self._device,
            verbose=False,
            stop_training=lambda: obj.budget_exceeded,
        )

        if obj.budget_exceeded:
            return

        # === Bayesian Optimization in Latent Space ===

        vae.eval()
        dtype = torch.float64

        # Define bounds for latent space
        z_bounds = torch.tensor([[-3.0] * latent_dim, [3.0] * latent_dim], dtype=dtype)
        unit_bounds = torch.stack(
            [
                torch.zeros(latent_dim, dtype=dtype),
                torch.ones(latent_dim, dtype=dtype),
            ],
            dim=0,
        )

        # Initialize with Sobol samples
        sobol = torch.quasirandom.SobolEngine(
            dimension=latent_dim, scramble=True, seed=random_seed
        )
        if obj.evals_left is not None:
            n_initial = min(n_initial, obj.evals_left)
        if n_initial < 1:
            return

        train_z_norm = sobol.draw(n=n_initial).to(dtype=dtype)
        train_z = unnormalize(train_z_norm, z_bounds)

        with torch.no_grad():
            train_x_decoded = vae.decode(train_z.float().to(self._device))

        # Evaluate initial samples through Objective
        train_x_jax = t2j(train_x_decoded.cpu())
        train_y = self._evaluate_candidates(obj, train_x_jax)

        train_y_torch = torch.from_numpy(np.array(train_y)).to(dtype=dtype)
        train_y_torch = train_y_torch.unsqueeze(-1)
        train_y_torch = -train_y_torch  # Negate for maximization

        # Filter out invalid values
        valid_mask = torch.isfinite(train_y_torch.squeeze())
        train_z_norm = train_z_norm[valid_mask]
        train_y_torch = train_y_torch[valid_mask]

        if len(train_y_torch) == 0:
            return

        # Acquisition optimization options
        acqf_options = {
            "raw_samples": acqf_raw_samples,
            "num_restarts": acqf_num_restarts,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }

        # BO loop - iterate up to max_iterations
        bo_iterations = 0

        while (
            max_iterations is None or bo_iterations < max_iterations
        ) and not obj.budget_exceeded:
            bo_iterations += 1

            # Fit GP model
            gp = SingleTaskGP(train_z_norm, train_y_torch)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            # Optimize acquisition function
            acqf = qLogEI(gp, train_y_torch.max())

            current_bo_batch_size = bo_batch_size
            if obj.evals_left is not None:
                current_bo_batch_size = min(current_bo_batch_size, obj.evals_left)
            if current_bo_batch_size < 1:
                break

            candidates_norm, _ = optimize_acqf(
                acqf,
                bounds=unit_bounds,
                q=current_bo_batch_size,
                gen_candidates=gen_candidates_scipy,
                **acqf_options,
            )

            # Unnormalize and decode
            candidates_z = unnormalize(candidates_norm, z_bounds)

            with torch.no_grad():
                candidates_x = vae.decode(candidates_z.float().to(self._device))

            # Evaluate through Objective
            candidates_x_jax = t2j(candidates_x.cpu())
            candidate_losses = self._evaluate_candidates(obj, candidates_x_jax)

            candidates_y = torch.from_numpy(-np.array(candidate_losses)).to(dtype=dtype)
            candidates_y = candidates_y.unsqueeze(-1)
            valid_mask = torch.isfinite(candidates_y.squeeze(-1))

            if not torch.any(valid_mask):
                continue

            candidates_norm = candidates_norm[valid_mask]
            candidates_y = candidates_y[valid_mask]

            # Update training data
            train_z_norm = torch.cat([train_z_norm, candidates_norm], dim=0)
            train_y_torch = torch.cat([train_y_torch, candidates_y], dim=0)
