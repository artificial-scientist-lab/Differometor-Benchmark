"""Example script to run TuRBO optimization on the Voyager problem."""

from dfbench import ConstrainedVoyagerProblem, BotorchTuRBO, VoyagerProblem
import jax.numpy as jnp

# Initialize the Voyager problem
vp = VoyagerProblem()

# Create TuRBO optimizer
optimizer = BotorchTuRBO(vp, max_iterations=100)

# Run optimization with Thompson Sampling acquisition
_, _, losses, wti = optimizer.optimize(
    save_to_file=True,
    wall_times=[10, 20, 30, 60, 120],
    batch_size=4,
    acqf="ts",
)

print("Best loss:", jnp.min(losses))
print("Wall time indices:", wti)
