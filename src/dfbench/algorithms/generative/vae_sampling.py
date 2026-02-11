"""VAE-based optimization with Bayesian Optimization in learned latent space.

This module implements a two-phase optimization approach:
1. VAE Training Phase: Trains a Variational Autoencoder on high-quality samples
   (either objective-guided or random) to learn a compressed latent representation
2. BO Phase: Performs Bayesian Optimization in the learned latent space using
   Gaussian Process surrogate models and batch Expected Improvement acquisition

The VAE learns a low-dimensional (d/10) latent space that captures the structure
of high-quality solutions, making optimization more efficient in high dimensions.
"""

import secrets

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Array, Float
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.optim import optimize_acqf
from botorch.generation import gen_candidates_scipy
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from dfbench.core.utils import t2j, j2t
from dfbench.core.problem import ContinuousProblem
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
) -> None:
    """Train the VAE with cyclic KL annealing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        cycle_len = 20
        beta = min(1.0, (epoch % cycle_len) / (cycle_len * 0.5))

        for batch in dataloader:
            x = batch[0].to(device)
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
    1. VAE Training Phase: Trains a Variational Autoencoder on high-quality samples
       (either objective-guided or random) to learn a compressed latent representation
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

    def __init__(self) -> None:
        """Initialize VAESampling optimizer.

        No algorithm-specific configuration needed at initialization.
        All parameters are passed to optimize().
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        vae_training_samples: int = 1000,
        vae_epochs: int = 100,
        batch_size: int = 64,
        latent_dim_factor: int = 10,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        use_objective_guidance: bool = True,
        top_k: int = 20,
        n_initial: int = 20,
        acqf_raw_samples: int = 512,
        acqf_num_restarts: int = 4,
    ) -> Objective:
        """Run VAE training followed by Bayesian Optimization in latent space.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of BO iterations in latent space (required).
            init_params: Initial parameters to seed optimization (unused).
            random_seed: Random seed for reproducibility.
            vae_training_samples: Number of samples to generate for VAE training.
            vae_epochs: Number of epochs to train VAE.
            batch_size: Batch size for evaluation and training.
            latent_dim_factor: Factor to determine latent dimension (d // latent_dim_factor + 1).
            hidden_dim: Hidden dimension for VAE architecture.
            num_blocks: Number of residual blocks in VAE.
            use_objective_guidance: If True, evaluate samples and train on top performers.
                                   If False, train on random samples without evaluation.
            top_k: Number of top-performing samples to select for VAE training
                during the objective-guided sampling phase.
            n_initial: Number of initial Sobol samples for BO phase.
            acqf_raw_samples: Number of raw samples for acquisition optimization.
            acqf_num_restarts: Number of restarts for acquisition optimization.

        Returns:
            The Objective instance with all logged data.
        """
        obj = problem_objective
        problem = obj.problem

        self.setup_objective(obj, unbounded=True, random_seed=random_seed)

        if random_seed is None:
            random_seed = secrets.randbits(32)
        obj.set_seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        print(f"Random seed: {random_seed}")

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

        obj.start_logging()

        # === VAE Training Phase ===
        if use_objective_guidance:
            # Mode 1: Objective-Guided Training
            # Generate candidates
            candidates_torch = (
                torch.randn(vae_training_samples, input_dim, device=self._device) * 1.65
            )

            # Evaluate candidates
            candidates_jax = t2j(candidates_torch.cpu())

            # Process in batches
            num_candidates = candidates_jax.shape[0]
            eval_losses_list = []
            for i in range(0, num_candidates, batch_size):
                batch = candidates_jax[i : i + batch_size]
                # Track evaluations through Objective (vmap_value handles all tracking)
                batch_losses = obj.vmap_value(batch)
                eval_losses_list.append(batch_losses)

            eval_losses = jnp.concatenate(eval_losses_list)

            # Select top performers
            top_indices = jnp.argsort(eval_losses)[:top_k]
            best_samples_jax = candidates_jax[top_indices]
            best_samples_torch = j2t(best_samples_jax).to(self._device)

            # Train VAE on best samples
            dataset = TensorDataset(best_samples_torch)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            vae.train()
            train_vae(vae, loader, epochs=vae_epochs, device=self._device, verbose=False)

        else:
            # Mode 2: Pure Random Sampling
            data = torch.randn(vae_training_samples, input_dim, device=self._device) * 1.65
            dataset = TensorDataset(data)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            vae.train()
            train_vae(vae, loader, epochs=vae_epochs, device=self._device, verbose=False)

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
        train_z_norm = sobol.draw(n=n_initial).to(dtype=dtype)
        train_z = unnormalize(train_z_norm, z_bounds)

        with torch.no_grad():
            train_x_decoded = vae.decode(train_z.float().to(self._device))

        # Evaluate initial samples through Objective
        train_x_jax = t2j(train_x_decoded.cpu())
        train_y_list = []
        for i in range(n_initial):
            loss_val = obj.value(train_x_jax[i])
            train_y_list.append(float(loss_val))

        train_y_torch = torch.tensor(train_y_list, dtype=dtype).unsqueeze(-1)
        train_y_torch = -train_y_torch  # Negate for maximization

        # Filter out invalid values
        valid_mask = torch.isfinite(train_y_torch.squeeze())
        train_z_norm = train_z_norm[valid_mask]
        train_y_torch = train_y_torch[valid_mask]

        if len(train_y_torch) == 0:
            return obj

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

        while (max_iterations is None or bo_iterations < max_iterations) and not obj.budget_exceeded:
            bo_iterations += 1

            # Fit GP model
            gp = SingleTaskGP(train_z_norm, train_y_torch)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            # Optimize acquisition function
            acqf = qLogEI(gp, train_y_torch.max())

            candidates_norm, _ = optimize_acqf(
                acqf,
                bounds=unit_bounds,
                q=1,
                gen_candidates=gen_candidates_scipy,
                **acqf_options,
            )

            # Unnormalize and decode
            candidates_z = unnormalize(candidates_norm, z_bounds)

            with torch.no_grad():
                candidates_x = vae.decode(candidates_z.float().to(self._device))

            # Evaluate through Objective
            candidates_x_jax = t2j(candidates_x.cpu())
            candidate_loss = obj.value(candidates_x_jax[0])

            if not jnp.isfinite(candidate_loss):
                continue

            candidates_y = torch.tensor(
                [-float(candidate_loss)], dtype=dtype
            ).unsqueeze(-1)

            # Update training data
            train_z_norm = torch.cat([train_z_norm, candidates_norm], dim=0)
            train_y_torch = torch.cat([train_y_torch, candidates_y], dim=0)

        return obj
