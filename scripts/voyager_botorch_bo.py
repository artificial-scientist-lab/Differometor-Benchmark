from dfbench import ConstrainedVoyagerProblem, BotorchBO
import jax.numpy as jnp
# Whole workflow of opimization with adam

vp = ConstrainedVoyagerProblem()

optimizer = BotorchBO(vp)

_, _, losses, wti = optimizer.optimize(
    save_to_file=True,
    wall_times=[10,20,30,60,120],
)

print("Best loss:", jnp.min(losses))
print("Wall time indices:", wti)
