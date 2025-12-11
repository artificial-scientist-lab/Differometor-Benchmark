from dfbench import VoyagerProblem, BotorchBO
import jax.numpy as jnp
# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = BotorchBO(vp)

_, _, losses, wti = optimizer.optimize(
    save_to_file=True,
    wall_times=[60, 120, 300, 600],
)

print("Best loss:", jnp.min(losses))
print("Wall time indices:", wti)
