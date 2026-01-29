"""VAE-based optimization with Bayesian Optimization in learned latent space.

This module implements a two-phase optimization approach:
1. VAE Training Phase: Trains a Variational Autoencoder on high-quality samples
   (either objective-guided or random) to learn a compressed latent representation
2. BO Phase: Performs Bayesian Optimization in the learned latent space using
   Gaussian Process surrogate models and batch Expected Improvement acquisition

The VAE learns a low-dimensional (d/10) latent space that captures the structure
of high-quality solutions, making optimization more efficient in high dimensions.
"""

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

from dfbench.core.utils import t2j_numpy as t2j, j2t_numpy as j2t
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
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
        _problem (ContinuousProblem): The optimization problem instance.
        _batch_size (int): Number of individuals to evaluate per batch.
        _hidden_dim (int): Hidden dimension for VAE architecture.
        _num_blocks (int): Number of residual blocks in VAE encoder/decoder.
        _use_objective_guidance (bool): Whether to train VAE on top performers.
        _device (torch.device): PyTorch device (cuda if available, else cpu).

    Note:
        This algorithm uses `problem.sigmoid_objective_function` which applies
        sigmoid bounding to handle infinite parameter spaces during VAE training.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = VAESampling(problem, batch_size=64, hidden_dim=256)
        >>> objective = optimizer.optimize(
        ...     max_time=120,
        ...     sampling_time_percentage=0.5,
        ... )
    """

    algorithm_str: str = "vae_sampling"
    algorithm_type: AlgorithmType = AlgorithmType.GENERATIVE

    def __init__(
        self,
        problem: ContinuousProblem,
        batch_size: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        use_objective_guidance: bool = True,
        verbose: int = 0,
        save_params_history: bool = True,
        save_batched_losses: bool = True,
        save_batched_params: bool = False,
    ) -> None:
        """Initialize VAESampling optimizer.

        Args:
            problem: The continuous optimization problem to solve.
            batch_size: Batch size for evaluation and training.
            hidden_dim: Hidden dimension for VAE.
            num_blocks: Number of residual blocks.
            use_objective_guidance: If True, evaluate samples and train on top performers.
                                   If False, train on random samples without evaluation.
            verbose (int): Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to True.
            save_batched_losses: Whether to save full batched losses (vs reduced).
                Defaults to True for detailed analysis.
            save_batched_params: Whether to save full batched params (memory heavy).
                Defaults to False.
        """
        self._problem = problem
        self._batch_size = batch_size
        self._hidden_dim = hidden_dim
        self._num_blocks = num_blocks
        self._use_objective_guidance = use_objective_guidance
        self._verbose = verbose
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._save_params_history = save_params_history
        self._save_batched_losses = save_batched_losses
        self._save_batched_params = save_batched_params

    def _vmap_sigmoid_objective(
        self, params_batch: Float[Array, "batch n_params"]
    ) -> Float[Array, "batch"]:
        """Vectorized sigmoid objective evaluation."""
        return jax.vmap(self._problem.sigmoid_objective_function)(params_batch)

    def optimize(
        self,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        random_seed: int | None = None,
        max_time: float | None = None,
        sampling_time_percentage: float = 0.5,
        top_k: int = 20,
        n_initial_bo: int = 20,
        verbose: int | None = None,
        print_every: int = 10,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        **vae_kwargs,
    ) -> Objective:
        """Run VAE training followed by Bayesian Optimization in latent space.

        Args:
            init_params: Initial parameters to seed optimization.
            random_seed: Random seed for reproducibility.
            max_time: Time budget in seconds. None for unlimited.
            sampling_time_percentage: Fraction of total max_time to allocate to
                VAE training phase. The remaining time is used for BO. For example, 0.5
                means 50% VAE training, 50% BO. Defaults to 0.5.
            top_k: Number of top-performing samples to select for VAE training
                during the objective-guided sampling phase. Defaults to 20.
            n_initial_bo: Number of initial Sobol samples for BO phase.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call obj.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            **vae_kwargs: Additional keyword arguments for VAE training.

        Returns:
            The Objective instance with all logged data.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        input_dim = self._problem.n_params
        latent_dim = input_dim // 10 + 1

        # Create VAE
        vae = ResNetVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=self._hidden_dim,
            num_res_blocks=self._num_blocks,
        )
        vae.to(self._device)

        # Create Objective wrapper (unbounded because we use sigmoid_objective_function)
        obj = Objective(
            self._problem,
            unbounded=True,  # Using sigmoid_objective_function for VAE phase
            max_time=max_time,
            max_evals=None,  # Time-based termination
            save_params_history=self._save_params_history,
            save_batched_losses_history=self._save_batched_losses,
            save_batched_history=self._save_batched_params,
            print_every=print_every,
            verbose=verbose if verbose is not None else self._verbose,
            algorithm_str=self.algorithm_str,
        )

        # Warmup JIT
        if self._verbose >= 1:
            print(f"Warming up JIT compilation...")
        _ = obj.value(jnp.zeros(input_dim))

        obj.start_logging()

        # === VAE Training Phase ===
        if verbose is None:
            verbose = self._verbose
        if verbose > 0:
            print(f"\n=== Starting VAE Training Phase ===")
            print(f"VAE phase: {sampling_time_percentage:.0%} of time budget")

        vae_iteration = 0
        while obj.time_progress_fraction < sampling_time_percentage and not obj.budget_exceeded:
            vae_iteration += 1

            if self._use_objective_guidance:
                # Mode 1: Objective-Guided Training
                if vae_iteration == 1:  # First iteration
                    candidates_torch = (
                        torch.randn(1000, input_dim, device=self._device) * 1.65
                    )
                else:
                    vae.eval()
                    with torch.no_grad():
                        z = torch.randn(
                            1000, vae.fc_mu.out_features, device=self._device
                        )
                        candidates_torch = vae.decode(z)

                # Evaluate candidates
                candidates_jax = t2j(candidates_torch.cpu())

                # Process in batches
                num_candidates = candidates_jax.shape[0]
                eval_losses_list = []
                for i in range(0, num_candidates, self._batch_size):
                    batch = candidates_jax[i : i + self._batch_size]
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
                train_vae(vae, loader, epochs=20, device=self._device, verbose=verbose > 1)

            else:
                # Mode 2: Pure Random Sampling
                data = torch.randn(1000, input_dim, device=self._device) * 1.65
                dataset = TensorDataset(data)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

                vae.train()
                train_vae(vae, loader, epochs=40, device=self._device, verbose=verbose > 1)

        if verbose > 0:
            print(f"\n=== VAE Training Complete ===")
            print(f"VAE phase used {obj.time_progress_fraction:.1%} of budget ({obj.time_elapsed:.1f}s)")

        # === Bayesian Optimization in Latent Space ===
        if verbose > 0:
            print("\n=== Starting BO in Latent Space ===")
            print(f"Remaining time budget: {obj.time_left:.1f}s" if obj.time_left else "Remaining time budget: unlimited")

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
        train_z_norm = sobol.draw(n=n_initial_bo).to(dtype=dtype)
        train_z = unnormalize(train_z_norm, z_bounds)

        with torch.no_grad():
            train_x_decoded = vae.decode(train_z.float().to(self._device))

        # Evaluate initial samples through Objective
        train_x_jax = t2j(train_x_decoded.cpu())
        train_y_list = []
        for i in range(n_initial_bo):
            loss_val = obj.value(train_x_jax[i])
            train_y_list.append(float(loss_val))

        train_y_torch = torch.tensor(train_y_list, dtype=dtype).unsqueeze(-1)
        train_y_torch = -train_y_torch  # Negate for maximization

        # Filter out invalid values
        valid_mask = torch.isfinite(train_y_torch.squeeze())
        train_z_norm = train_z_norm[valid_mask]
        train_y_torch = train_y_torch[valid_mask]

        if len(train_y_torch) == 0:
            if verbose > 0:
                print("Warning: All initial BO evaluations returned NaN/Inf.")
            if plot_loss:
                obj.output_to_files()
            if save_run_to_file:
                obj.save_run_data()
            return obj

        # Acquisition optimization options
        acqf_options = {
            "raw_samples": 512,
            "num_restarts": 4,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }

        # BO loop - use budget_exceeded which checks max_time
        bo_iterations = 0

        while not obj.budget_exceeded:
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

        if verbose > 0:
            print(f"\n=== BO Complete ===")
            print(f"Total BO iterations: {bo_iterations}")

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj
