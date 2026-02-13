# Problems

All optimization problems in dfbench represent gravitational-wave detector design tasks. The goal is to find optical component parameters (mirror reflectivities, laser power, cavity lengths, etc.) that minimize the detector's strain sensitivity across a frequency range of 20–5000 Hz.

**Import:**

```python
from dfbench.problems import VoyagerProblem, ConstrainedVoyagerProblem, RandomUIFOProblem
```

---

## Problem Hierarchy

```
ContinuousProblem          (ABC — core/problem.py)
  └── OpticalSetupProblem  (ABC — problems/base_problem.py)
        ├── VoyagerProblem
        ├── ConstrainedVoyagerProblem
        └── RandomUIFOProblem
```

### `ContinuousProblem` (Abstract Base)

Defines the minimal interface every problem must implement:

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `name` | `str` | Human-readable identifier. |
| `objective_function` | `Callable` | Loss in bounded parameter space. |
| `sigmoid_objective_function` | `Callable` | Loss in unbounded space (sigmoid bounding applied internally). |
| `bounds` | `Array[2, n_params]` | `[lower_bounds, upper_bounds]` for each parameter. |
| `optimization_pairs` | `list[tuple[str, str]]` | `(component_name, property_name)` tuples mapping each parameter index to a Differometor component. |
| `n_params` | `int` | Number of parameters = `len(optimization_pairs)`. |

**Rationale — two objective functions:** See the [Architecture Overview](Architecture-Overview#1-problem-layer) for why both bounded and unbounded variants exist.

### `OpticalSetupProblem` (Optical Base)

Extends `ContinuousProblem` with optics-specific functionality shared by all Differometor problems:

- **Frequency grid:** A log-spaced array of frequencies from 20 Hz to 5 kHz (configurable via `n_frequencies`).
- **Target sensitivity:** Stored in `_target_sensitivities`, computed from the reference detector design at initialization.
- **`calculate_sensitivity(params)`:** Computes the sensitivity curve for a given parameter vector — used for plotting, not optimization.
- **`output_to_files(…)`:** Writes JSON parameter/loss files and PNG plots (loss curve + sensitivity curve vs. target).

---

## Available Problems

### `VoyagerProblem`

| Property | Value |
|----------|-------|
| Setup | LIGO Voyager with balanced homodyne detection |
| Parameters | ~25 (reflectivities, tunings, squeezing, power, masses, lengths, phases) |
| Noise model | Single quantum noise source |
| Speed | ~12 ms/eval on A100 GPU |
| Difficulty | Moderate — loss < 0 achievable without physical constraints |

```python
problem = VoyagerProblem(n_frequencies=100)
```

#### How the loss works

1. The Voyager reference setup is simulated to get a target sensitivity curve.
2. The target loss is $\sum \log_{10}(\text{sensitivity}_{\text{target}})$.
3. For a candidate parameter set, the loss is:
   $$\text{loss} = \sum \log_{10}(\text{sensitivity}_{\text{candidate}}) - \text{target\_loss}$$
4. **Loss < 0** means the candidate has better sensitivity than the reference design.

**Rationale for log-scale loss:** Strain sensitivities span many orders of magnitude ($10^{-20}$ to $10^{-24}$). Summing log-sensitivities treats improvements at all frequencies equally rather than being dominated by the worst band.

#### What the parameters represent

Each parameter corresponds to a `(component, property)` pair from the Voyager setup:

| Property | Bounds | Physical meaning |
|----------|--------|-----------------|
| `reflectivity` | [0, 1] | Fraction of light reflected by a mirror |
| `tuning` | [0, 90] | Phase tuning of a mirror in degrees |
| `db` | [0.01, 20] | Squeezing level in decibels |
| `angle` | [-180, 180] | Squeezing angle in degrees |
| `power` | [0.01, 200] | Laser power in watts |
| `mass` | [0.01, 200] | Mirror suspension mass in kg |
| `length` | [1, 4000] | Cavity arm length in meters |
| `phase` | [-180, 180] | Phase offset in degrees |

#### Caveat

`VoyagerProblem` does **not** enforce physical constraints (e.g. maximum mirror power absorption). A solution with loss < 0 may be physically unrealizable. For constrained optimization, use `ConstrainedVoyagerProblem`.

---

### `ConstrainedVoyagerProblem`

| Property | Value |
|----------|-------|
| Setup | LIGO Voyager with balanced homodyne detection |
| Parameters | ~25 (same as `VoyagerProblem`) |
| Noise model | Three sources: quantum, amplitude, frequency noise |
| Constraints | Power thresholds on mirrors, beamsplitters, and detectors |
| Speed | ~25 ms/eval on A100 GPU |
| Difficulty | Hard — loss < 0 is very difficult to achieve |

```python
problem = ConstrainedVoyagerProblem(n_frequencies=100)
```

#### Differences from `VoyagerProblem`

1. **Realistic noise model:** Uses three separate modulation modes (quantum noise, amplitude noise, frequency noise) and combines their contributions into a single sensitivity curve. This produces a more accurate picture of real detector performance.

2. **Power constraints:** The loss includes a penalty term for violating optical power limits:
   - `HARD_SIDE_POWER_THRESHOLD` — maximum power on mirror/beamsplitter side ports
   - `SOFT_SIDE_POWER_THRESHOLD` — softer limit with gradual penalty
   - `DETECTOR_POWER_THRESHOLD` — maximum power on detector ports

   The penalty is passed through $p / (1 + p)$ to squash it into $[0, 1)$, preventing it from dominating the loss while still providing a smooth gradient signal.

**Rationale — penalty squashing:** A raw penalty can become orders of magnitude larger than the sensitivity loss, making gradient-based optimizers ignore sensitivity entirely. The $p/(1+p)$ transform bounds the penalty contribution while preserving its gradient direction.

---

### `RandomUIFOProblem`

| Property | Value |
|----------|-------|
| Setup | Quasi-Universal Interferometer (UIFO) with random topology |
| Parameters | 50–250+ depending on grid size |
| Noise model | Three sources (same as constrained Voyager) |
| Constraints | Power thresholds (same as constrained Voyager) |
| Speed | ~500 ms/eval on A100 GPU |
| Difficulty | Hard but achievable — the UIFO is overparameterized |

```python
problem = RandomUIFOProblem(size=3, n_frequencies=100, topology_seed=42)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | `3` | Grid dimensions (3 = 3×3). Larger grids have more components and parameters. |
| `n_frequencies` | `100` | Frequency points for sensitivity calculation. |
| `topology_seed` | `42` | Seed for the random graph structure. The same seed always produces the same interferometer topology. |

#### What is a UIFO?

A Quasi-Universal Interferometer Field Optimization (UIFO) is a grid-based interferometer where beamsplitters, mirrors, lasers, and squeezers are placed on a grid and connected by spaces. The topology (which components are placed where and how they connect) is generated randomly from `topology_seed`. Once the topology is fixed, only the continuous parameters (reflectivities, tunings, lengths, etc.) are optimized.

**Rationale — coupled grid-cell spaces:** Horizontal and vertical spaces at the same grid positions are constrained to have equal lengths via `constrain_inter_grid_cell_spaces()`. This preserves the physical grid structure and prevents the optimizer from "folding" the interferometer into a degenerate geometry.

#### Design note

The reference sensitivity target is always the Voyager detector. Since the UIFO is overparameterized (many more degrees of freedom than Voyager), it can in principle achieve better sensitivity but the large parameter space makes optimization harder.

---
