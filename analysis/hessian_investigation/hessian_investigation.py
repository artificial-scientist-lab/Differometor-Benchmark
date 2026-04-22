# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import jax.numpy as jnp
from differometor.setups import voyager
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench.algorithms import AdamGD
from dfbench import Objective
from dfbench.core.utils import inverse_sigmoid_bounding
from differometor.utils import sigmoid_bounding
from differometor.simulate import simulate
from differometor.utils import calculate_sensitivities, calculate_powers
import jax
import matplotlib.pyplot as plt
from jax import config
import jax.random as jr
import numpy as np

config.update("jax_enable_x64", True)

# %%
S, component_property_pairs = voyager()

# Extract bounded parameter values from the Voyager setup S,
# using the same optimization pairs and bounds that VoyagerProblem uses
vp = ConstrainedVoyagerProblem()

voyager_params = []
for component_name, property_name in vp.optimization_pairs:
    if "_" not in component_name:
        value = S.nodes[component_name]["properties"][property_name]
    else:
        value = S.edges[component_name]["properties"][property_name]
    voyager_params.append(value)

# voyager_params[i] corresponds to vp.optimization_pairs[i]
voyager_params = jnp.array(voyager_params)


print(f"Number of parameters: {len(voyager_params)}")
print(f"Bounded params (first 5): {voyager_params[:5]}")

# %%
# Sanity check: evaluate the objective with the Voyager params
# Loss should be ~0 since these are the target params
obj = Objective(vp, unbounded=False, max_evals=20000, max_time=300, verbose=1)
loss = obj.value(voyager_params)
print(f"Loss at Voyager params: {loss}")
print("(Should be ~0 since these are the target parameters)")

# %% [markdown]
# ### Explore Surface Around Voyager Parameters

# %%
# Decompose the loss at the Voyager params to see what's happening
q_results = simulate(**{**vp._q_arrays, "optimized_parameters": voyager_params})
ampl_results = simulate(**{**vp._ampl_arrays, "optimized_parameters": voyager_params})
freq_results = simulate(**{**vp._freq_arrays, "optimized_parameters": voyager_params})
results = [
    (*q_results, *vp._q_metadata),
    (*ampl_results, *vp._ampl_metadata),
    (*freq_results, *vp._freq_metadata),
]

sensitivities = calculate_sensitivities(
    results, vp._sensitivity_function, vp._frequencies, homodyne=True
)
powers = calculate_powers(q_results[0], *vp._q_metadata)

sensitivity_loss, penalty, violations = vp._calculate_loss(
    sensitivities, vp._target_sensitivities, powers
)

print(f"Sensitivity loss: {sensitivity_loss}")
print(f"Penalty (raw):    {penalty}")
print(f"Penalty (scaled): {penalty / (1 + penalty)}")
print(f"Total loss:       {sensitivity_loss + penalty / (1 + penalty)}")
print(f"\nViolations: {violations}")
print(f"Max violation:    {jnp.max(violations)}")

# Now check the gradient magnitude and direction
grad_fn = jax.grad(vp.objective_function)
grads = grad_fn(voyager_params)
print(f"\nGradient norm:    {jnp.linalg.norm(grads):.6e}")
print(f"Max |grad|:       {jnp.max(jnp.abs(grads)):.6e}")
print(f"Grad (first 5):   {grads[:5]}")

# Show which parameters have the largest gradients
top_indices = jnp.argsort(jnp.abs(grads))[-10:]
print("\nTop 10 parameters by gradient magnitude:")
for idx in reversed(top_indices):
    name = vp.optimization_pairs[int(idx)]
    print(
        f"  {name}: grad={grads[int(idx)]:.6e}, value={voyager_params[int(idx)]:.6f}, bounds=[{vp.bounds[0, int(idx)]}, {vp.bounds[1, int(idx)]}]"
    )

# %%
# 1D sweep along each parameter axis from the Voyager point
# For each parameter, evaluate loss at small perturbations while holding others fixed
obj_fn = vp.objective_function
n_params = len(voyager_params)
n_steps = 50  # points per side (total 2*n_steps + 1 per parameter)

# Sweep a small fraction of each parameter's range
sweep_fraction = 0.001  # 0.1% of range in each direction

results = {}
for idx in range(n_params):
    name = vp.optimization_pairs[idx]
    param_range = vp.bounds[1, idx] - vp.bounds[0, idx]
    center = voyager_params[idx]
    delta = sweep_fraction * param_range

    # Sweep points, clipped to bounds
    offsets = jnp.linspace(-delta, delta, 2 * n_steps + 1)
    values = jnp.clip(center + offsets, vp.bounds[0, idx], vp.bounds[1, idx])

    losses = []
    for val in values:
        test_params = voyager_params.at[idx].set(val)
        l = obj_fn(test_params)
        losses.append(float(l))

    losses = jnp.array(losses)
    min_loss = jnp.min(losses)
    min_idx = jnp.argmin(losses)
    min_val = float(values[min_idx])

    results[idx] = {
        "name": name,
        "center": float(center),
        "values": values,
        "losses": losses,
        "min_loss": float(min_loss),
        "min_val": min_val,
    }

# Print summary: which parameters have loss < 0 somewhere?
print("Parameters where loss < 0 is achievable with single-parameter perturbation:")
print("=" * 80)
negative_params = []
for idx in range(n_params):
    r = results[idx]
    if r["min_loss"] < -1e-10:
        negative_params.append(idx)
        print(
            f"  {r['name']}: min_loss={r['min_loss']:.6e} at value={r['min_val']:.6f} (Voyager={r['center']:.6f})"
        )

if not negative_params:
    print("  NONE — no single-parameter perturbation achieves loss < 0")

print(f"All parameters sorted by minimum achievable loss:")
print("-" * 80)
sorted_indices = sorted(range(n_params), key=lambda i: results[i]["min_loss"])
for idx in sorted_indices[:15]:  # show top 15
    r = results[idx]
    print(
        f"  {r['name']}: min_loss={r['min_loss']:.6e}, Voyager={r['center']:.6f}, best_at={r['min_val']:.6f}"
    )

# %%
# Plot the 1D loss sweeps for the 12 most sensitive parameters
sorted_indices = sorted(range(n_params), key=lambda i: results[i]["min_loss"])
plot_indices = sorted_indices[:12]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for ax_idx, param_idx in enumerate(plot_indices):
    r = results[param_idx]
    ax = axes[ax_idx]
    ax.plot(r["values"], r["losses"], "b-", linewidth=1.5)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="loss=0")
    ax.axvline(x=r["center"], color="g", linestyle="--", alpha=0.5, label="Voyager")
    comp, prop = r["name"]
    ax.set_title(f"{comp}.{prop}", fontsize=10)
    ax.set_ylabel("Loss")
    ax.tick_params(labelsize=8)
    if ax_idx == 0:
        ax.legend(fontsize=7)

fig.suptitle("1D loss sweeps around Voyager parameters (0.1% of range)", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Plot 1D loss sweeps for the three parameters with largest gradients:
# etmy.reflectivity, etmx.reflectivity, fm2.reflectivity
target_names = [
    ("etmy", "reflectivity"),
    ("etmx", "reflectivity"),
    ("fm2", "reflectivity"),
]
target_indices = [
    idx for idx in range(n_params) if results[idx]["name"] in target_names
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax_idx, param_idx in enumerate(target_indices):
    r = results[param_idx]
    ax = axes[ax_idx]
    ax.plot(r["values"], r["losses"], "b-", linewidth=1.5)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="loss=0")
    ax.axvline(x=r["center"], color="g", linestyle="--", alpha=0.5, label="Voyager")
    comp, prop = r["name"]
    ax.set_title(f"{comp}.{prop}", fontsize=10)
    ax.set_ylabel("Loss")
    ax.tick_params(labelsize=8)
    if ax_idx == 0:
        ax.legend(fontsize=7)

fig.suptitle("1D loss sweeps for high-gradient parameters (0.1% of range)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Hessian Exploration

# %%
obj_fn = jax.jit(vp.objective_function)
obj_grad = jax.jit(jax.grad(obj_fn))
obj_hessian = jax.jit(jax.hessian(obj_fn))

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm


def plot_hessian(hess, labels, name, title_suffix="", save=False):
    """Plot Hessian analysis: linear, log, and eigendecomposition.

    Args:
        hess: Hessian matrix (n x n).
        labels: List of parameter labels.
        name: String prefix for saved PDF filenames.
        title_suffix: Optional string appended to plot titles.
    """
    title_tag = f" ({title_suffix})" if title_suffix else ""

    # 1. Linear scale
    vmax = max(abs(hess.min()), abs(hess.max()))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.matshow(hess, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(
        im,
        ax=ax,
        shrink=0.8,
        label="$\\partial^2 L / \\partial \\theta_i \\partial \\theta_j$",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Hessian{title_tag}")
    plt.tight_layout()
    if save:
        plt.savefig(f"{name}_hessian.pdf")
    plt.show()

    # 2. sign * log1p scale (adapts to any range automatically)
    hess_log = jnp.sign(hess) * jnp.log1p(jnp.abs(hess))
    vmax_log = max(abs(hess_log.min()), abs(hess_log.max()))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.matshow(hess_log, cmap="RdBu_r", vmin=-vmax_log, vmax=vmax_log)
    fig.colorbar(
        im,
        ax=ax,
        shrink=0.8,
        label="$\\mathrm{sgn}(H_{ij}) \\cdot \\log(1 + |H_{ij}|)$",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Hessian{title_tag} (log scale)")
    plt.tight_layout()
    if save:
        plt.savefig(f"{name}_hessian_log.pdf")
    plt.show()

    # 3. Eigendecomposition
    D, U = jnp.linalg.eigh(hess)
    n = len(D)

    fig, (ax_u, ax_d) = plt.subplots(
        2,
        1,
        figsize=(14, 14),
        gridspec_kw={"height_ratios": [4, 1.8]},
        sharex=True,
    )

    # Eigenvectors: values in [-1, 1]
    umax = max(abs(U.min()), abs(U.max()))
    im = ax_u.imshow(
        U,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1e-2, vmin=-umax, vmax=umax),
        aspect="auto",
        extent=[-0.5, n - 0.5, n - 0.5, -0.5],
        interpolation="nearest",
    )

    divider = make_axes_locatable(ax_u)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(im, cax=cax, label="Eigenvector component")
    ax_u.set_yticks(range(len(labels)))
    ax_u.set_yticklabels(labels, fontsize=7)
    ax_u.set_ylabel("Parameter")
    ax_u.set_title(f"Eigenvectors (U) and eigenvalues (D){title_tag}")
    ax_u.tick_params(bottom=False, labelbottom=False)

    # Eigenvalues: raw values with symlog y-scale
    colors = ["tab:red" if d > 0 else "tab:blue" for d in D]
    ax_d.bar(range(n), D, color=colors, width=0.8)
    ax_d.set_yscale("symlog", linthresh=1e-6)
    ax_d.set_ylim(min(1.5 * D.min(), 0.0), max(1.5 * D.max(), 0.0))
    ax_d.set_ylabel("Eigenvalue")
    ax_d.set_xlabel("Eigenvector index")
    ax_d.axhline(0, color="k", linewidth=0.5)
    ax_d.set_xlim(-0.5, n - 0.5)

    divider_d = make_axes_locatable(ax_d)
    cax_d = divider_d.append_axes("right", size="2%", pad=0.1)
    cax_d.set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(f"{name}_hessian_eigendecomp.pdf")
    plt.show()

    # Print summary
    n_neg = int((D < 0).sum())
    n_zero = int((jnp.abs(D) < 1e-6).sum())
    n_pos = int((D > 0).sum())
    print(
        f"Eigenvalue summary: {n_neg} negative, {n_zero} near-zero (<1e-6), {n_pos} positive"
    )
    print(f"Range: [{float(D.min()):.4e}, {float(D.max()):.4e}]")
    print(f"Sorted Eigvals: {jnp.sort(D)}")
    print(f"Condition number: {jnp.linalg.cond(hess):.4e}")

    return D, U


# %%
lower, upper = vp.bounds[0], vp.bounds[1]
rand_points = lower + (upper - lower) * jr.uniform(jr.PRNGKey(42), (3, 48))

# %%
adam_params_zero = jnp.load(
    "adam_constrained_voyager_seed_73_lr_0.1_max_evals_20000_arcsinh_transform.npz"
)["best_constrained_params"]  # Params found with Adam with loss of 0.0066302
print(f"Loss at Adam params: {obj.value(adam_params_zero):.6e}")

# %%
voyager_hess = obj_hessian(voyager_params).block_until_ready()
rand_zero_hess = obj_hessian(rand_points[0]).block_until_ready()
rand_one_hess = obj_hessian(rand_points[1]).block_until_ready()
rand_two_hess = obj_hessian(rand_points[2]).block_until_ready()
adam_zero_hess = obj_hessian(adam_params_zero).block_until_ready()

# %%
print(f"Objective value at Voyager params: {obj.value(voyager_params):.6e}")
print(f"Objective value at rand_0 params: {obj.value(rand_points[0]):.6e}")
print(f"Objective value at rand_1 params: {obj.value(rand_points[1]):.6e}")
print(f"Objective value at rand_2 params: {obj.value(rand_points[2]):.6e}")
print(f"Objective value at Adam params: {obj.value(adam_params_zero):.6e}")

# %%
labels = [f"{c}.{p}" for c, p in vp.optimization_pairs]
D, U = plot_hessian(
    voyager_hess,
    labels,
    name="voyager",
    title_suffix="at voyager params",
    save=False,
)

# %% [markdown]
# ### Whitening Transformation

# %%
# Eigendecomposition of the Voyager Hessian
D, U = jnp.linalg.eigh(voyager_hess)
grad_at_voyager = obj_grad(voyager_params)

# Index of smallest absolute eigenvalue (flattest direction)
flat_idx = int(jnp.argmin(jnp.abs(D)))
print(f"Flattest eigenvector: index {flat_idx}, λ = {float(D[flat_idx]):.4e}")

# %%
# 1D slices of the objective along selected eigenvectors
x0 = jnp.array(voyager_params)
g = jnp.array(grad_at_voyager)
L0 = obj_fn(x0)
lower, upper = vp.bounds[0], vp.bounds[1]

eigvec_indices = [0, flat_idx, -1]  # most negative, flattest, largest positive
zoom_levels = [3, 0.1, 0.01]

for idx in eigvec_indices:
    direction = U[:, idx]
    lam = float(D[idx])
    scale = 1.0 / jnp.sqrt(jnp.abs(D[idx]))

    # Compute alpha where bounds are first hit in each direction
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios_upper = np.where(
            direction > 0, (upper - x0) / direction, (lower - x0) / direction
        )
        ratios_lower = np.where(
            direction > 0, (lower - x0) / direction, (upper - x0) / direction
        )
    alpha_max = float(
        jnp.min(jnp.where(jnp.isfinite(ratios_upper), ratios_upper, jnp.inf))
    )
    alpha_min = float(
        jnp.max(jnp.where(jnp.isfinite(ratios_lower), ratios_lower, -jnp.inf))
    )
    print(
        f"Eigvec {idx}: λ={lam:.4e}, scale={scale:.4e}, "
        f"bounds at [{alpha_min / scale:.15f}, {alpha_max / scale:.15f}] char. lengths"
    )

    fig, axes = plt.subplots(len(zoom_levels), 1, figsize=(12, 4 * len(zoom_levels)))

    for ax, z in zip(axes, zoom_levels):
        alphas = jnp.linspace(-z * scale, z * scale, 200)
        losses = jnp.array([obj_fn(x0 + alpha * direction) for alpha in alphas])
        quad = L0 + alphas * (g @ direction) + 0.5 * alphas**2 * lam

        ax.plot(alphas / scale, losses, label="Objective", linewidth=2)
        ax.plot(
            alphas / scale,
            quad,
            "--",
            label="Quadratic approx",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)

        # Mark bound limits
        for a_bound, label in [(alpha_min, "lower bound"), (alpha_max, "upper bound")]:
            a_scaled = a_bound / scale
            if -z <= a_scaled <= z:
                ax.axvline(
                    a_scaled,
                    color="red",
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                    label=label,
                )

        ax.set_xlabel(
            r"$\alpha \cdot \sqrt{|\lambda|}$  (units of characteristic length)"
        )
        ax.set_ylabel("Objective value")
        ax.set_title(f"±{z} char. lengths")
        ax.legend()
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle(f"Objective along eigenvector {idx} (λ = {lam:.4e})", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"voyager_1d_slice_eigvec_{idx}.pdf", dpi=300)
    plt.show()

# %% [markdown]
# ### Preconditioning the Gradient Visualisation

# %%
grad_at_voyager = obj_grad(voyager_params)
hess_at_voyager = obj_hessian(voyager_params)
preconditioned_grad_at_voyager = jnp.linalg.solve(hess_at_voyager, grad_at_voyager)

# %%
grad_at_voyager

# %%
preconditioned_grad_at_voyager

# %%
import numpy as np

raw_grad = np.asarray(grad_at_voyager, dtype=float)
precond_grad = np.asarray(preconditioned_grad_at_voyager, dtype=float)
labels = [f"{c}.{p}" for c, p in vp.optimization_pairs]

raw_unit = raw_grad / (np.linalg.norm(raw_grad) + 1e-18)
precond_unit = precond_grad / (np.linalg.norm(precond_grad) + 1e-18)
cosine_similarity = float(np.dot(raw_unit, precond_unit))

n = len(labels)
y = np.arange(n)

fig, (ax_raw, ax_pre) = plt.subplots(1, 2, figsize=(14, 12), sharey=True)

ax_raw.barh(y, raw_grad, height=0.7, color="steelblue")
ax_raw.set_xscale("symlog", linthresh=1e-6)
ax_raw.set_yticks(y)
ax_raw.set_yticklabels(labels, fontsize=7)
ax_raw.invert_yaxis()
ax_raw.axvline(0, color="black", linewidth=0.5)
ax_raw.set_xlabel("Component value (symlog)")
ax_raw.set_title("Raw gradient")
ax_raw.grid(axis="x", alpha=0.25, linestyle="--")

ax_pre.barh(y, precond_grad, height=0.7, color="indianred")
ax_pre.set_xscale("symlog", linthresh=1e-6)
ax_pre.invert_yaxis()
ax_pre.axvline(0, color="black", linewidth=0.5)
ax_pre.set_xlabel("Component value (symlog)")
ax_pre.set_title(r"Preconditioned gradient ($H^{-1}g$)")
ax_pre.grid(axis="x", alpha=0.25, linestyle="--")

fig.suptitle(
    f"Raw vs preconditioned gradient at Voyager parameters\n"
    f"cos(g, H⁻¹g) = {cosine_similarity:.4f}",
    fontsize=13,
)
plt.tight_layout()
plt.show()

print(f"||g||₂      = {np.linalg.norm(raw_grad):.4e}")
print(f"||H⁻¹g||₂   = {np.linalg.norm(precond_grad):.4e}")
print(f"cos(g, H⁻¹g) = {cosine_similarity:.6f}")

# %%
# Damped Hessian preconditioning via eigenvalue clamping
H = np.asarray(hess_at_voyager, dtype=float)
g = np.asarray(grad_at_voyager, dtype=float)

D_eig, U_eig = np.linalg.eigh(H)

# Clamp: use |eigenvalue|, floored at epsilon to avoid blowup
epsilon = np.max(np.abs(D_eig)) * 1e-6
D_clamped = np.sign(D_eig) * np.maximum(np.abs(D_eig), epsilon)
D_safe = np.maximum(np.abs(D_eig), epsilon)  # all-positive version (guaranteed descent)

damped_grad = U_eig @ (U_eig.T @ (-g) / D_clamped)
safe_grad = U_eig @ (U_eig.T @ (-g) / D_safe)

labels = [f"{c}.{p}" for c, p in vp.optimization_pairs]
n = len(labels)
y = np.arange(n)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


fig, (ax_raw, ax_clamp, ax_safe) = plt.subplots(1, 3, figsize=(18, 12), sharey=True)

ax_raw.barh(y, -g, height=0.7, color="steelblue")
ax_raw.set_xscale("symlog", linthresh=1e-6)
ax_raw.set_yticks(y)
ax_raw.set_yticklabels(labels, fontsize=7)
ax_raw.invert_yaxis()
ax_raw.axvline(0, color="black", linewidth=0.5)
ax_raw.set_xlabel("Component value (symlog)")
ax_raw.set_title(f"Steepest descent (−g)\ncos(−g, −g) = 1.000")
ax_raw.grid(axis="x", alpha=0.25, linestyle="--")

ax_clamp.barh(y, damped_grad, height=0.7, color="mediumpurple")
ax_clamp.set_xscale("symlog", linthresh=1e-6)
ax_clamp.invert_yaxis()
ax_clamp.axvline(0, color="black", linewidth=0.5)
ax_clamp.set_xlabel("Component value (symlog)")
cos_clamp = cos_sim(-g, damped_grad)
ax_clamp.set_title(f"Clamped eigenvalues\ncos(−g, step) = {cos_clamp:.4f}")
ax_clamp.grid(axis="x", alpha=0.25, linestyle="--")

ax_safe.barh(y, safe_grad, height=0.7, color="indianred")
ax_safe.set_xscale("symlog", linthresh=1e-6)
ax_safe.invert_yaxis()
ax_safe.axvline(0, color="black", linewidth=0.5)
ax_safe.set_xlabel("Component value (symlog)")
cos_safe = cos_sim(-g, safe_grad)
ax_safe.set_title(f"|eigenvalue| clamped (descent)\ncos(−g, step) = {cos_safe:.4f}")
ax_safe.grid(axis="x", alpha=0.25, linestyle="--")

fig.suptitle(
    f"Newton step with damped Hessian (ε = max|λ| × 1e-6)",
    fontsize=13,
)
plt.tight_layout()
plt.show()

print(f"||−g||₂           = {np.linalg.norm(g):.4e}")
print(f"||clamped step||₂  = {np.linalg.norm(damped_grad):.4e}")
print(f"||safe step||₂     = {np.linalg.norm(safe_grad):.4e}")

# %%
# Diagnostics: check whether the Newton direction is a descent direction here.
g = jnp.asarray(grad_at_voyager)
H = jnp.asarray(hess_at_voyager)
Hg = jnp.asarray(preconditioned_grad_at_voyager)  # solves H @ x = g
d_newton = -Hg

eps = 1e-18
g_norm = float(jnp.linalg.norm(g))
Hg_norm = float(jnp.linalg.norm(Hg))
d_norm = float(jnp.linalg.norm(d_newton))

cos_g_Hinv_g = float((g @ Hg) / ((jnp.linalg.norm(g) * jnp.linalg.norm(Hg)) + eps))
cos_g_newton = float(
    (g @ d_newton) / ((jnp.linalg.norm(g) * jnp.linalg.norm(d_newton)) + eps)
)

# For minimization, we want g^T d_newton < 0.
gTd = float(g @ d_newton)
gTHinv_g = float(g @ Hg)

sym_err = float(jnp.linalg.norm(H - H.T) / (jnp.linalg.norm(H) + eps))
eigvals = jnp.linalg.eigvalsh((H + H.T) / 2.0)
eigvals_np = np.asarray(eigvals)
n_neg = int((eigvals_np < -1e-10).sum())
n_zero = int((np.abs(eigvals_np) <= 1e-10).sum())
n_pos = int((eigvals_np > 1e-10).sum())
cond_H = float(np.linalg.cond(np.asarray(H)))

print("Gradient/Hessian diagnostics at Voyager parameters")
print("=" * 64)
print(f"||g||_2               : {g_norm:.6e}")
print(f"||H^-1 g||_2          : {Hg_norm:.6e}")
print(f"||d_newton||_2        : {d_norm:.6e}")
print(f"cos(g, H^-1 g)        : {cos_g_Hinv_g:.6f}")
print(f"cos(g, d_newton)      : {cos_g_newton:.6f}")
print(f"g^T d_newton          : {gTd:.6e}")
print(f"g^T H^-1 g            : {gTHinv_g:.6e}")
print(f"relative symmetry err : {sym_err:.3e}")
print(f"eig sign counts       : neg={n_neg}, near0={n_zero}, pos={n_pos}")
print(f"cond(H)               : {cond_H:.6e}")

if gTd < 0:
    print("\nInterpretation: Newton direction is locally a descent direction.")
else:
    print(
        "\nInterpretation: Newton direction is NOT descent here (indefinite/unstable local Hessian likely)."
    )

if n_neg > 0:
    print("Hessian appears indefinite (has negative eigenvalues).")
elif n_zero > 0:
    print("Hessian appears singular/near-singular in some directions.")
else:
    print("Hessian appears positive definite at this point.")

# %% [markdown]
# ### Newton optimisation

# %%
import time
import numpy as np

lower, upper = vp.bounds[0], vp.bounds[1]
init_params = voyager_params.copy()

# JIT-compiled functions
obj_fn = jax.jit(vp.objective_function)
obj_val_grad = jax.jit(jax.value_and_grad(vp.objective_function))
obj_hessian = jax.jit(jax.hessian(vp.objective_function))

eval_count = 0
loss_history = []
t0 = time.time()


def projected_newton_line_search(
    x0, max_iter=10, alpha_init=1.0, alpha_min=1e-16, c=1e-4, rho=0.5
):
    """Newton's method with backtracking line search and bound projection.

    At each iteration:
    1. Identify free variables (not at bounds with gradient pointing outward)
    2. Solve Newton system in the free subspace
    3. Backtracking line search with Armijo condition, projecting onto bounds
    """
    global eval_count
    x = jnp.array(x0, dtype=jnp.float64)

    for iteration in range(max_iter):
        loss, g = obj_val_grad(x)
        eval_count += 1
        loss_history.append(float(loss))
        elapsed = time.time() - t0

        grad_norm = float(jnp.linalg.norm(g))
        print(
            f"  iter {iteration:4d} | loss: {float(loss):.6e} | "
            f"|grad|: {grad_norm:.2e} | {elapsed:.1f}s"
        )

        if grad_norm < 1e-12:
            print("  Converged: gradient norm below tolerance")
            break

        # Identify free variables: not at a bound with gradient pushing outward
        at_lower = jnp.abs(x - lower) < 1e-15
        at_upper = jnp.abs(x - upper) < 1e-15
        clamped = (at_lower & (g > 0)) | (at_upper & (g < 0))
        free = ~clamped

        n_free = int(free.sum())
        if n_free == 0:
            print("  Converged: all variables at bounds (KKT satisfied)")
            break

        # Compute Hessian
        H = obj_hessian(x)

        # Extract free subspace
        free_idx = jnp.where(free)[0]
        H_free = H[jnp.ix_(free_idx, free_idx)]
        g_free = g[free_idx]

        # Solve Newton system in free subspace: H_free @ d_free = -g_free
        # Use regularisation if H is singular/indefinite
        try:
            d_free = jnp.linalg.solve(H_free, -g_free)
        except Exception:
            # Fall back to gradient descent if solve fails
            d_free = -g_free

        # Check if Newton direction is a descent direction
        directional_deriv = float(g_free @ d_free)
        if directional_deriv > 0:
            # Newton direction is ascending — use negative gradient instead
            print("Newton is ascending, switching to negative gradient direction")
            d_free = -g_free
            directional_deriv = float(g_free @ d_free)

        # Build full step direction
        d = jnp.zeros_like(x)
        d = d.at[free_idx].set(d_free)

        # Backtracking line search with Armijo condition and bound projection
        alpha = alpha_init
        while alpha > alpha_min:
            x_new = jnp.clip(x + alpha * d, lower, upper)
            loss_new = obj_fn(x_new)
            eval_count += 1
            loss_history.append(float(loss_new))

            # Armijo condition: sufficient decrease
            actual_step = x_new - x
            predicted_decrease = float(g @ actual_step)
            if float(loss_new) <= float(loss) + c * predicted_decrease:
                print(f"Alpha used: {alpha:.2e}")
                break
            alpha *= rho
        else:
            print(f"  Line search failed at iter {iteration} (alpha={alpha:.2e})")
            break

        x = x_new
        print(
            f"         alpha: {alpha:.2e} | free: {n_free}/{len(x)} | "
            f"step norm: {float(jnp.linalg.norm(actual_step)):.2e}"
        )

    return x


print(f"Initial loss: {float(obj_fn(init_params)):.6e}")
eval_count = 0  # reset after warmup call
loss_history = []
print(f"Parameters: {len(init_params)}")
print()

newton_best_params = projected_newton_line_search(init_params)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"Final loss:  {float(obj_fn(newton_best_params)):.6e}")
print(f"Evaluations: {eval_count}")

# Show which parameters changed most
param_range = upper - lower
diffs = newton_best_params - init_params
rel_diffs = diffs / param_range
top_changed = jnp.argsort(jnp.abs(rel_diffs))[-10:]
print(f"\nTop 10 parameters by relative change:")
for idx in reversed(top_changed):
    idx = int(idx)
    name = vp.optimization_pairs[idx]
    print(
        f"  {name}: {float(init_params[idx]):.6f} -> {float(newton_best_params[idx]):.6f} "
        f"(rel: {float(rel_diffs[idx]):.6e})"
    )

# %%
# Plot loss curve
fig, ax = plt.subplots(figsize=(10, 4))
plot_losses = [l for l in loss_history if l < 1e9]
ax.plot(plot_losses, linewidth=0.8)
ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
ax.set_xlabel("Function evaluation (finite only)")
ax.set_ylabel("Loss")
ax.set_title(f"Newton — final loss: {float(obj_fn(newton_best_params)):.6e}")
plt.tight_layout()
plt.show()

# %%
newton_best_params

# %%
voyager_params

# %%
newton_best_params[8]

# %%
voyager_params[8]

# %%
vp.optimization_pairs[8]

# %%
vp.optimization_pairs
