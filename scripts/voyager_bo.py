from dfbench import VoyagerProblem, BayesianOptimization
import jax.numpy as jnp
# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = BayesianOptimization(vp)

_, _, losses, wti = optimizer.optimize(
    save_to_file=True,
    wall_times=[30, 60, 900, 1800],
)

print("Best loss:", jnp.min(losses))
print("Wall time indices:", wti)
