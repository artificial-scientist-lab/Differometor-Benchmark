# Optimized Campaign H5 Format (Canonical Schema)

This file defines the desired on-disk H5 structure for optimized campaigns.

This is a schema/spec document, not a tutorial.
Writers (`optimize.py`, `simplify.py`, `reevaluate*`) should only persist keys described here.

## 1. Root Layout

```text
/
  /experiment_data
  /runs
    /<run_id>
      ... run payload ...
```

- `/experiment_data`: experiment-level metadata shared by all runs.
- `/runs/<run_id>`: one optimized run payload.

## 2. Run Entry Schema (Authoritative)

Each `/runs/<run_id>` entry is a JSON-like object with these allowed keys.

### 2.1 Required Run Keys

| Key | Type | Meaning |
|---|---|---|
| `setup_graph` | `str` or `dict` | Serialized setup graph for the run. |
| `best_parameters` | `list[float]` or `list[list[float]]` | Best parameter vector/matrix for the run. |
| `complexity` | scalar int-like | Graph complexity metric for the run. |
| `loss*_senspow` | scalar float-like | Required senspow loss key (for example `loss_senspow` or `loss_incoherent_senspow`). |
| `sensitivities_<RANGE>` | list-like numeric | At least one sensitivity key is required. |
| `power_data_<RANGE>` | object | At least one power payload key is required (must follow section 2.3). |

### 2.2 Optional Run Keys

| Key Pattern | Type | Meaning |
|---|---|---|
| `loss_*` | scalar float-like | Any additional run-level loss (for example `loss_optimized`). At least one `loss*_senspow` key is required (section 2.1). |
| `signal_transfer_<RANGE>` | list-like numeric | Optional coherent/q-only detector signal-transfer curve for one range. |
| `signal_transfer_incoherent_<RANGE>` | list-like numeric | Optional incoherent detector signal-transfer curve for one range. |
| `noises` | object | Optional per-range noise-curve payload; see section 2.5. |
| `simplifications` | object | Nested simplification payloads, see section 3. |

No other run-level keys are allowed.

### 2.3 `power_data_<RANGE>` Payload Schema

Each `power_data_<RANGE>` object must contain exactly:

| Key | Type | Meaning |
|---|---|---|
| `raw_powers` | list-like numeric | Power values for the range. |
| `power_port_to_index` | object | Mapping from canonical port name to column index in `raw_powers` (see section 2.3.1). |
| `violating` | bool-like or list-like bool | Violation marker(s) for the same range. |

No other keys are allowed inside `power_data_<RANGE>`.

#### 2.3.1 `power_port_to_index` Port Name Requirements

- Keys must use canonical dotted port names such as:
  - `mt21.left.in`
  - `center11.top.out`
  - `boundary20bhbs.right.in`
- Keys must not use abstract side placeholders like:
  - `hard_side_*`
  - `soft_side_*`
- Each key maps to the column index in `raw_powers`.

### 2.4 Signal-Transfer Curve Schema (Optional)

Each `signal_transfer_<RANGE>` or `signal_transfer_incoherent_<RANGE>` value is
a list-like numeric curve for the same frequency samples as the corresponding
sensitivity and noise curves.

The curve stores the absolute detector signal-transfer magnitude used to
normalize noise curves into equivalent-input sensitivity.

### 2.5 `noises` Payload Schema (Optional)

`noises` supports both the canonical raw-noise layout written by current
reevaluation scripts and the legacy flat layout present in existing
`qamplfreq_flat` files.

#### 2.5.1 Canonical Raw-Noise Layout

The canonical layout is keyed first by range name. Each range stores one shared
raw quantum curve plus optional model-specific raw noise buckets:

```json
{
  "broadband": {
    "quantum": [...],
    "coherent": {
      "coherent_laser_amplitude": [...],
      "coherent_laser_frequency": [...],
      "total": [...]
    },
    "incoherent": {
      "incoherent_laser_amplitude": [...],
      "incoherent_laser_frequency": [...],
      "seismic": [...],
      "thermal": [...],
      "classical_total": [...],
      "total": [...]
    }
  }
}
```

Rules:

- Range keys must be strings (for example `broadband`, `post_merger`).
- Each range payload must contain shared raw `quantum`.
- Noise-model keys must be `coherent` or `incoherent`.
- Curves are raw unnormalized amplitude spectral density/noise-amplitude curves.
  Equivalent-input sensitivity can be restored from raw noise and the matching
  signal-transfer curve.
- Allowed curve keys:
  - `quantum` (range-level shared raw quantum noise)
  - `coherent_laser_amplitude`, `coherent_laser_frequency` (`coherent`)
  - `incoherent_laser_amplitude`, `incoherent_laser_frequency`, `seismic`, `thermal` (`incoherent`)
  - `classical_total` (`incoherent`)
  - `total` (quadrature total for the stored noise-model payload)
- Curve values must be list-like numeric.

#### 2.5.2 Legacy Flat Noise Layout

Many existing files in `/home/phylomatx/Documents/qamplfreq_flat/` currently
store noise curves directly under the range, without `coherent` or `incoherent`
buckets:

```json
{
  "broadband": {
    "quantum": [...],
    "incoherent_laser_amplitude": [...],
    "incoherent_laser_frequency": [...],
    "seismic": [...],
    "thermal": [...]
  }
}
```

Some existing flat payloads also include coherent laser curves in the same
range payload:

```json
{
  "broadband": {
    "quantum": [...],
    "coherent_laser_amplitude": [...],
    "coherent_laser_frequency": [...],
    "incoherent_laser_amplitude": [...],
    "incoherent_laser_frequency": [...],
    "seismic": [...],
    "thermal": [...]
  }
}
```

This flat layout is accepted for compatibility until files are backfilled with
the canonical raw-noise layout.

Flat legacy curves are equivalent-input curves that were already normalized by
the detector signal transfer. They are not raw noise curves and should not be
combined with `signal_transfer_<RANGE>` to reconstruct sensitivity.

The interpretation of `noises/<RANGE>/quantum` depends on the surrounding
layout:

- If the same range payload contains `coherent` or `incoherent` buckets, the
  range-level `quantum` is the canonical shared raw quantum noise curve.
- If the range payload contains only flat curve keys and no model bucket,
  `quantum` is part of the legacy equivalent-input payload.

### 2.6 Explicitly Forbidden Run Keys

These must not be persisted at `/runs/<run_id>`:

- `initialized_from` (experiment-level only)
- `optimization_pairs`
- `fractional_length_couplings`
- `bounds` (must be summarized at experiment-level in `parameter_type_bounds`)
- `initial_guess`
- `tunable_mask`
- `centers`
- `boundaries`
- `reported_loss`
- `reevaluated_loss_senspow`
- training/debug/history keys such as `losses`, `best_loss`, `best_loss_no_violations`, `convergence_time`, `iteration_time`, `step`, `final`, etc.

## 3. Simplification Payload Schema

Path:

```text
/runs/<run_id>/simplifications/<strategy>/<threshold>
```

Each payload allows only:

### 3.1 Required Simplification Keys

| Key | Type | Meaning |
|---|---|---|
| `setup_graph` | `str` or `dict` | Simplified setup graph. |
| `best_parameters` | `list[float]` or `list[list[float]]` | Parameters for simplified setup. |
| `complexity` | scalar int-like | Graph complexity metric for the simplified setup. |
| `loss*_senspow` | scalar float-like | Required senspow loss key (for example `loss_senspow` or `loss_incoherent_senspow`). |
| `sensitivities_<RANGE>` | list-like numeric | At least one sensitivity key is required. |
| `power_data_<RANGE>` | object | At least one power payload key is required (must follow section 2.3). |

### 3.2 Optional Simplification Keys

| Key Pattern | Type | Meaning |
|---|---|---|
| `loss_*` | scalar float-like | Additional simplification loss values; at least one `loss*_senspow` key is required (section 3.1). |
| `signal_transfer_<RANGE>` or `signal_transfer_incoherent_<RANGE>` | list-like numeric | Optional detector signal-transfer curve; same schema as section 2.4. |
| `noises` | object | Optional per-range noise-curve payload; same schema as section 2.5. |

### 3.3 Simplification Power Payload Schema

`power_data_<RANGE>` payloads under simplifications must use the same schema as section 2.3.

### 3.4 Forbidden Simplification Keys

- `bounds`
- `frequency_values`
- `loss` (legacy key)
- `reported_loss`
- `reevaluated_loss_senspow`
- `optimization_pairs`
- `fractional_length_couplings`

## 4. Experiment Data Schema

`/experiment_data` stores experiment-level metadata.

### 4.1 Required for Script Interoperability

| Key | Type | Meaning |
|---|---|---|
| `optimized_properties` | list[str] | Used to infer `optimization_pairs` from setup graph. |
| `parameter_type_bounds` | object | Parameter-type to `[low, high]` bounds mapping. |

`uifo_length_coupling` is optional. If absent, scripts must assume:

- `uifo_length_coupling = "inter_cell_equal"`

### 4.2 Common Additional Keys

`optimize.py` writes full experiment args/config here (plus metadata), including commonly:

- `frequency_values`
- `frequency_ranges`
- `qamplfreq`
- penalty/regularization settings
- topology/optimizer settings
- `initialized_from` (if warm-started)

`initialized_from` belongs here (experiment-level), not at run level.

## 5. Inference Rules (No Run-Level Pair Metadata)

Scripts must infer `optimization_pairs` (and fractional coupling rules) from:

- run `setup_graph`
- experiment `optimized_properties`
- experiment `uifo_length_coupling` (default to `inter_cell_equal` if missing)

Therefore, run entries must not store `optimization_pairs` or `fractional_length_couplings`.

## 6. Canonical Run Examples

### 6.1 Minimal Valid Run

```json
{
  "setup_graph": "{...}",
  "best_parameters": [[...]],
  "complexity": 12,
  "loss_senspow": 0.123,
  "sensitivities_post_merger": [...],
  "power_data_post_merger": {
    "raw_powers": [...],
    "power_port_to_index": {"mt21.left.in": 0, "mt21.left.out": 1},
    "violating": false
  }
}
```

### 6.2 Typical Valid Run with Range Data and Simplifications

```json
{
  "setup_graph": "{...}",
  "best_parameters": [[...]],
  "complexity": 14,
  "loss_optimized": 0.123,
  "loss_senspow": 0.127,
  "sensitivities_post_merger": [...],
  "power_data_post_merger": {
    "raw_powers": [...],
    "power_port_to_index": {"mt21.left.in": 0, "mt21.left.out": 1},
    "violating": false
  },
  "simplifications": {
    "closeness_with_skip": {
      "0.5": {
        "setup_graph": "{...}",
        "best_parameters": [[...]],
        "complexity": 9,
        "sensitivities_post_merger": [...],
        "power_data_post_merger": {
          "raw_powers": [...],
          "power_port_to_index": {"mt21.left.in": 0, "mt21.left.out": 1},
          "violating": false
        },
        "loss_senspow": 0.14
      }
    }
  }
}
```

## 7. Compatibility Notes

- Legacy key `reevaluated_loss_senspow` should be normalized to a canonical `loss*_senspow` key.
- Legacy `reported_loss` should be removed.
- Legacy key `powers` in power payloads should be normalized to `raw_powers`.
- Legacy simplification keys `loss` and `frequency_values` should be removed.
- Numeric storage dtypes can vary (`float32/float64`, `int32/int64`).
