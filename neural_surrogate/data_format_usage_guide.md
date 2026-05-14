# Optimized Campaign H5: Minimal Standalone Usage (Old `differometor` Compatible)

This guide is intentionally minimal.
It shows how to render a setup from campaign `.h5` data using an older
`differometor` checkout with:

- no `scripts/h5_helper.py`
- no `scripts/helper.py`
- no reevaluation/resimulation

## 1. Requirements

- Python with `h5py` and `numpy`.
- A checkout of the old `differometor` repo that provides:
  - `differometor.setups.Setup.from_data`
  - `differometor.plot.visualize_setup`

## 2. Data Structure (What Is In The H5)

At top level, each file has:

- `/experiment_data`: campaign-level metadata and defaults.
- `/runs`: one group per run id (`"0"`, `"1"`, ...), each containing run payloads.

Minimal structure you will typically use:

```text
/
  experiment_data/
    frequency_ranges
    frequency_values
    optimized_properties
    parameter_type_bounds
    ...
  runs/
    <run_id>/
      setup_graph
      best_parameters
      complexity
      loss_*
      sensitivities_<RANGE>
      signal_transfer_<RANGE>
      signal_transfer_incoherent_<RANGE>
      noises/
        <RANGE>/
          quantum
          coherent/
            coherent_laser_amplitude
            coherent_laser_frequency
            total
          incoherent/
            incoherent_laser_amplitude
            incoherent_laser_frequency
            seismic
            thermal
            classical_total
            total
          (or current legacy flat curves directly under <RANGE>)
      power_data_<RANGE>/
        raw_powers
        power_port_to_index
        violating
      simplifications/
        <strategy>/
          <threshold>/
            (same shape as run payload)
```

Field meaning (practical):

- `experiment_data.frequency_ranges`: ordered range names (for example `["broadband", "post_merger"]`).
- `experiment_data.frequency_values`: sampled frequency grid used in the campaign.
- `experiment_data.optimized_properties`: parameter families optimized in this campaign.
- `experiment_data.parameter_type_bounds`: optimization bounds per parameter family.
- `runs/<run_id>/setup_graph`: serialized setup graph; pass to `Setup.from_data(...)`.
- `runs/<run_id>/best_parameters`: best parameter vector for the run.
- `runs/<run_id>/complexity`: run complexity score.
- `runs/<run_id>/loss_*`: one or more scalar losses (`loss_senspow`, `loss_incoherent_senspow`, etc.).
- `runs/<run_id>/sensitivities_<RANGE>`: sensitivity curve payload for one range.
- `runs/<run_id>/sensitivities_incoherent_<RANGE>`: incoherent sensitivity curve payload when available.
- `runs/<run_id>/signal_transfer_<RANGE>`: coherent/q-only detector signal-transfer curve for one range.
- `runs/<run_id>/signal_transfer_incoherent_<RANGE>`: incoherent detector signal-transfer curve when available.
- `runs/<run_id>/noises/<RANGE>/quantum`: shared raw quantum noise curve when `coherent` or `incoherent` buckets are present; in current flat legacy files it is an equivalent-input legacy curve.
- `runs/<run_id>/noises/<RANGE>/<noise_model>`: model-specific raw unnormalized noise curves; coherent and incoherent buckets can coexist.
- Current pre-backfill `qamplfreq_flat` files commonly store flat legacy curves directly as `runs/<run_id>/noises/<RANGE>/quantum`, `.../incoherent_laser_amplitude`, `.../incoherent_laser_frequency`, `.../seismic`, and `.../thermal`.
- `runs/<run_id>/power_data_<RANGE>/raw_powers`: stored powers array for plotting (no resimulation needed).
- `runs/<run_id>/power_data_<RANGE>/power_port_to_index`: mapping from canonical port name to index in `raw_powers`.
- `runs/<run_id>/power_data_<RANGE>/violating`: whether stored powers violate configured limits.
- `runs/<run_id>/simplifications/<strategy>/<threshold>`: simplified payload at strategy + threshold; same key schema as main run payload.

Notes:

- `<RANGE>` suffixes are dynamic and come from `frequency_ranges`.
- Some keys are optional depending on what was stored for that campaign.
- For visualization from stored powers, the required minimum is:
  - `setup_graph`
  - `power_data_<RANGE>.raw_powers`
  - `power_data_<RANGE>.power_port_to_index`

## 3. Example: Filter Broadband Runs And Display Losses

This example is analysis-only (no visualization). It scans all `.h5` files in a
directory, keeps only runs that contain broadband payloads, extracts a scalar
loss for each run, and prints a sorted ranking.

Broadband filter used in this example:

- run payload has `power_data_broadband`, or
- run payload has `sensitivities_broadband`

Loss extraction used in this example:

- prefer `loss_senspow`
- else `loss_incoherent_senspow`
- else `loss_optimized`
- else first scalar `loss_*` key found

Save as `broadband_loss_report.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr_json_type(node: h5py.Dataset) -> str | None:
    if "__json_type__" not in node.attrs:
        return None
    value = _decode_scalar(node.attrs["__json_type__"])
    return str(value) if value is not None else None


def _dataset_scalar(node: h5py.Dataset) -> Any:
    json_type = _attr_json_type(node)
    raw = node[()]

    if json_type == "null":
        return None

    value = raw
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
        else:
            return None

    value = _decode_scalar(value)
    if isinstance(value, str) and json_type == "json_blob":
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, (int, float, bool)) else None
    return value


def _get_scalar_loss(run_group: h5py.Group) -> tuple[float | None, str | None]:
    preferred = ("loss_senspow", "loss_incoherent_senspow", "loss_optimized")
    for key in preferred:
        if key not in run_group:
            continue
        node = run_group[key]
        if not isinstance(node, h5py.Dataset):
            continue
        value = _dataset_scalar(node)
        if isinstance(value, (int, float, np.number)):
            return float(value), key

    for key in run_group.keys():
        if not str(key).startswith("loss_"):
            continue
        node = run_group[key]
        if not isinstance(node, h5py.Dataset):
            continue
        value = _dataset_scalar(node)
        if isinstance(value, (int, float, np.number)):
            return float(value), str(key)
    return None, None


def _has_broadband(run_group: h5py.Group) -> bool:
    return ("power_data_broadband" in run_group) or ("sensitivities_broadband" in run_group)


def main() -> None:
    parser = argparse.ArgumentParser(description="List broadband runs and losses from campaign H5 files.")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing .h5 files.")
    parser.add_argument("--top-k", type=int, default=20, help="How many best runs to print.")
    args = parser.parse_args()

    h5_files = sorted(args.dataset_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in {args.dataset_dir}")

    rows: list[tuple[float, str, str, str]] = []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as handle:
            runs_group = handle.get("runs")
            if runs_group is None:
                continue
            run_ids = sorted(runs_group.keys(), key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
            for run_id in run_ids:
                node = runs_group[run_id]
                if not isinstance(node, h5py.Group):
                    continue
                if not _has_broadband(node):
                    continue
                loss_value, loss_key = _get_scalar_loss(node)
                if loss_value is None or loss_key is None:
                    continue
                rows.append((loss_value, h5_path.name, str(run_id), loss_key))

    rows.sort(key=lambda r: r[0])
    print(f"Broadband runs with scalar loss: {len(rows)}")
    print("rank\tloss\tloss_key\tfile\trun_id")
    for rank, (loss_value, file_name, run_id, loss_key) in enumerate(rows[: max(0, args.top_k)], start=1):
        print(f"{rank}\t{loss_value:.9g}\t{loss_key}\t{file_name}\t{run_id}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python broadband_loss_report.py /path/to/h5_dataset --top-k 30
```

## 4. Example: Visualize Setup From Stored Powers (No Resimulation)

This script is fully self-contained and does not depend on any repo helper.
It has one job: read one run payload (or one simplification payload) from H5,
extract `setup_graph` + stored powers, and render HTML via old `differometor`.

What the code in this section does:

- Decode H5 JSON-like payloads:
  - `read_h5_json(...)` recursively decodes groups/datasets back into Python objects.
  - It supports writer-specific dataset types via `__json_type__`:
    - `null`
    - `json_blob`
    - `string`
    - `port_index_table`
    - `compressed_utf8_blob`
  - It also preserves original dict keys using `__original_key__` when present.
- Load the target run:
  - `_load_experiment_and_run(...)` reads `/experiment_data` and `/runs/<run_id>`.
  - It validates that both required root groups exist and that run id resolves.
- Optionally switch to a simplification payload:
  - `_select_simplification_payload(...)` selects `simplifications/<strategy>/<threshold>`.
  - It accepts exact threshold keys and float-near matches (tiny tolerance).
- Resolve frequency range for power lookup:
  - `_resolve_range_name(...)` chooses `<RANGE>` for `power_data_<RANGE>`.
  - Priority is explicit `--range`, then first available from `frequency_ranges`, then first available power key.
- Extract visualization inputs:
  - `_extract_setup(...)` reads `setup_graph` and builds `Setup` via `Setup.from_data(...)`.
  - `_extract_power_payload(...)` reads `raw_powers` and `power_port_to_index`.
- Render without resimulation:
  - `visualize_setup(setup, powers=raw_powers, port_to_index=...)` generates the HTML plot directly from stored powers.
  - No reevaluation (`evaluate_setups`, `sensitivity_q_noise`, or added noise sources) is called.

Runtime flow in `main()`:

1. Parse CLI args (`h5_file`, `run_id`, optional simplification/range/output).
2. Decode `/experiment_data` and `/runs/<run_id>`.
3. Pick run payload or simplification payload.
4. Resolve range and fetch stored power payload.
5. Build setup object from `setup_graph`.
6. Call `visualize_setup(...)` and print resolved metadata.

Save as `visualize_setup_from_h5_standalone.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from differometor.plot import visualize_setup
from differometor.setups import Setup


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_python(item) for item in value.tolist()]
    if isinstance(value, np.void) and value.dtype.names:
        return {name: _to_python(value[name]) for name in value.dtype.names}
    return _decode_scalar(value)


def _attr_str(node: Any, name: str) -> str | None:
    if name not in node.attrs:
        return None
    value = _decode_scalar(node.attrs[name])
    return str(value) if value is not None else None


def _read_port_index_table(node: h5py.Dataset) -> dict[str, int]:
    table = node[()]
    if not isinstance(table, np.ndarray) or table.dtype.names is None:
        raise RuntimeError(f"port_index_table dataset is not structured: {node.name}")
    if "port" not in table.dtype.names or "index" not in table.dtype.names:
        raise RuntimeError(f"port_index_table dataset missing required fields: {node.name}")

    out: dict[str, int] = {}
    for row in table:
        port_raw = row["port"]
        idx_raw = row["index"]
        if isinstance(port_raw, (bytes, np.bytes_)):
            port = bytes(port_raw).rstrip(b"\x00").decode("utf-8", errors="replace")
        else:
            port = str(_decode_scalar(port_raw))
        out[port] = int(_decode_scalar(idx_raw))
    return out


def _read_compressed_utf8_blob(node: h5py.Dataset) -> str:
    raw = node[()]
    if isinstance(raw, np.ndarray):
        payload = raw.tobytes()
    elif isinstance(raw, (bytes, bytearray)):
        payload = bytes(raw)
    else:
        payload = bytes(np.asarray(raw, dtype=np.uint8).tobytes())
    return zlib.decompress(payload).decode("utf-8")


def read_h5_json(node: Any) -> Any:
    if isinstance(node, h5py.Group):
        json_type = _attr_str(node, "__json_type__")
        if json_type == "list":
            keys = sorted(node.keys(), key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
            return [read_h5_json(node[key]) for key in keys]

        out: dict[str, Any] = {}
        for key in node.keys():
            child = node[key]
            original_key = _attr_str(child, "__original_key__")
            out[original_key or key] = read_h5_json(child)
        return out

    if isinstance(node, h5py.Dataset):
        json_type = _attr_str(node, "__json_type__")
        raw = node[()]

        if json_type == "null":
            return None
        if json_type == "json_blob":
            return json.loads(_decode_scalar(raw))
        if json_type == "string":
            return str(_decode_scalar(raw))
        if json_type == "port_index_table":
            return _read_port_index_table(node)
        if json_type == "compressed_utf8_blob":
            return _read_compressed_utf8_blob(node)

        return _to_python(raw)

    raise TypeError(f"Unsupported HDF5 node type: {type(node)!r}")


def _load_experiment_and_run(h5_path: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    with h5py.File(h5_path, "r") as handle:
        if "experiment_data" not in handle or "runs" not in handle:
            raise RuntimeError("H5 file must contain /experiment_data and /runs.")

        experiment_data = read_h5_json(handle["experiment_data"])
        if not isinstance(experiment_data, dict):
            raise RuntimeError("Decoded /experiment_data is not a dict.")

        runs_group = handle["runs"]
        run_id_text = str(run_id)
        if run_id_text not in runs_group:
            numeric = str(int(run_id_text)) if run_id_text.isdigit() else None
            if numeric is not None and numeric in runs_group:
                run_id_text = numeric
            else:
                available = sorted(str(k) for k in runs_group.keys())
                raise RuntimeError(f"Run id '{run_id}' not found. Available: {available[:20]}")

        run_payload = read_h5_json(runs_group[run_id_text])
        if not isinstance(run_payload, dict):
            raise RuntimeError(f"Decoded /runs/{run_id_text} is not a dict.")

    return experiment_data, run_payload, run_id_text


def _as_range_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, (bytes, bytearray)):
                out.append(item.decode("utf-8", errors="replace"))
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _select_simplification_payload(
    run_payload: dict[str, Any],
    *,
    strategy: str,
    threshold: float,
) -> tuple[dict[str, Any], str]:
    simplifications = run_payload.get("simplifications")
    if not isinstance(simplifications, dict):
        raise RuntimeError("Run has no 'simplifications' payload.")

    strategy_bucket = simplifications.get(strategy)
    if not isinstance(strategy_bucket, dict):
        available = sorted(str(name) for name in simplifications.keys())
        raise RuntimeError(f"No strategy '{strategy}'. Available: {available}")

    requested = float(threshold)
    direct_keys = (f"{requested:.12g}", str(requested))
    for key in direct_keys:
        payload = strategy_bucket.get(key)
        if isinstance(payload, dict):
            return payload, str(key)

    tolerance = max(1e-9, abs(requested) * 1e-9)
    best_key: str | None = None
    best_payload: dict[str, Any] | None = None
    best_diff = math.inf
    for key, payload in strategy_bucket.items():
        if not isinstance(payload, dict):
            continue
        try:
            candidate = float(key)
        except Exception:
            continue
        diff = abs(candidate - requested)
        if diff <= tolerance and diff < best_diff:
            best_diff = diff
            best_key = str(key)
            best_payload = payload

    if best_payload is not None and best_key is not None:
        return best_payload, best_key

    available = sorted(str(k) for k in strategy_bucket.keys())
    raise RuntimeError(
        f"No simplification payload for strategy '{strategy}' at threshold {requested:.12g}. "
        f"Available keys: {available}"
    )


def _available_power_ranges(payload: dict[str, Any]) -> list[str]:
    ranges = []
    for key in payload.keys():
        key_text = str(key)
        if key_text.startswith("power_data_"):
            ranges.append(key_text.removeprefix("power_data_"))
    return sorted(ranges)


def _resolve_range_name(
    *,
    payload: dict[str, Any],
    experiment_data: dict[str, Any],
    requested_range: str | None,
) -> str:
    if requested_range is not None:
        selected = str(requested_range)
        if f"power_data_{selected}" not in payload:
            available = _available_power_ranges(payload)
            raise RuntimeError(f"Missing power_data_{selected}. Available: {available}")
        return selected

    experiment_ranges = _as_range_names(experiment_data.get("frequency_ranges"))
    for range_name in experiment_ranges:
        if f"power_data_{range_name}" in payload:
            return range_name

    available = _available_power_ranges(payload)
    if available:
        return available[0]

    raise RuntimeError("No power_data_<RANGE> payload found.")


def _extract_setup(payload: dict[str, Any]) -> Setup:
    setup_raw = payload.get("setup_graph")
    if isinstance(setup_raw, str):
        setup_raw = json.loads(setup_raw)
    if not isinstance(setup_raw, dict):
        raise RuntimeError("setup_graph must decode to a dict.")
    return Setup.from_data(setup_raw)


def _extract_power_payload(payload: dict[str, Any], range_name: str) -> tuple[Any, dict[str, Any]]:
    power_key = f"power_data_{range_name}"
    power_payload = payload.get(power_key)
    if not isinstance(power_payload, dict):
        raise RuntimeError(f"{power_key} missing or not a dict.")

    raw_powers = power_payload.get("raw_powers")
    if raw_powers is None:
        raise RuntimeError(f"{power_key}.raw_powers is missing.")

    port_to_index = power_payload.get("power_port_to_index")
    if not isinstance(port_to_index, dict):
        raise RuntimeError(f"{power_key}.power_port_to_index missing or not a dict.")

    return raw_powers, port_to_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render setup HTML from H5 payload + stored powers (no resimulation)."
    )
    parser.add_argument("h5_file", type=Path, help="Path to one .h5 file.")
    parser.add_argument("run_id", help="Run id under /runs (for example: 0).")
    parser.add_argument("--output-html", type=Path, default=Path("setup_from_h5.html"))
    parser.add_argument("--range", type=str, default=None, help="Range name (for example: broadband).")
    parser.add_argument("--simplification-strategy", type=str, default=None)
    parser.add_argument("--simplification-threshold", type=float, default=None)
    args = parser.parse_args()

    experiment_data, run_payload, resolved_run_id = _load_experiment_and_run(args.h5_file, str(args.run_id))

    source = "run"
    selected_payload = run_payload
    if args.simplification_strategy is not None or args.simplification_threshold is not None:
        if args.simplification_strategy is None or args.simplification_threshold is None:
            raise RuntimeError("Provide both --simplification-strategy and --simplification-threshold, or neither.")
        selected_payload, matched_key = _select_simplification_payload(
            run_payload,
            strategy=str(args.simplification_strategy),
            threshold=float(args.simplification_threshold),
        )
        source = f"simplification:{args.simplification_strategy}/{matched_key}"

    range_name = _resolve_range_name(
        payload=selected_payload,
        experiment_data=experiment_data,
        requested_range=args.range,
    )
    raw_powers, port_to_index = _extract_power_payload(selected_payload, range_name)
    setup = _extract_setup(selected_payload)

    visualize_setup(
        setup,
        output_file=str(args.output_html),
        powers=raw_powers,
        port_to_index=port_to_index,
    )

    print(f"Rendered setup to: {args.output_html.resolve()}")
    print(f"Resolved run id: {resolved_run_id}")
    print(f"Source payload: {source}")
    print(f"Power range: {range_name}")
    print(f"Mapped ports: {len(port_to_index)}")


if __name__ == "__main__":
    main()
```
