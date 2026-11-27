"""Compare RNG timings against the optional flagos-optest baseline file."""

import json
import math
import re
import warnings
from pathlib import Path

_RNG_SHAPE = re.compile(r"(?:\[(\d+)\]|\((\d+),\))(?:;lambda=([\d.eE+-]+))?")
_RESOURCES = {"mem": "bandwidth_gbps", "cuda": "vector", "tensor": "tensor"}
_DTYPES = {"torch.float32": "fp32", "torch.float64": "fp64", "torch.int32": "int32"}


def _identity(key):
    match = _RNG_SHAPE.fullmatch(key.replace(" ", ""))
    if match:
        n, lam = int(match[1] or match[2]), match[3]
        return n, float(lam) if lam else None
    return None


def _factor(scale, vendor, chip, unit, dtype):
    resource = _RESOURCES.get(unit)
    if not resource:
        return None
    vendor = {"ascend-cann900": "ascend", "ppu": "thead"}.get(vendor, vendor)
    for device in scale.get("chips", []):
        if device["vendor"].lower().replace("t-head", "thead") != vendor:
            continue
        if not re.search(rf"(?<![\w]){re.escape(device['chip'])}(?![\w])", chip, re.I):
            continue
        for source in ("measured", "nominal"):
            val = (device.get(source) or {}).get(resource)
            if isinstance(val, dict):
                val = val.get(_DTYPES.get(dtype))
            if isinstance(val, (int, float)) and val > 0 and math.isfinite(val):
                return val
    return None


def apply_external_baseline(operator_id, details, path, vendor, chip):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected an object keyed by operator ID")
    except (OSError, ValueError) as exc:
        warnings.warn(f"Cannot read --base-data {path}: {exc}. Keeping the measured native ratio.", stacklevel=2)
        for group in details:
            for point in group["result"]:
                point["base_miss_reason"] = "base_data_unavailable"
        return
    shapes = data.get(operator_id, {}).get("shapes", {})
    by_case = {}
    for key, typed in shapes.items():
        identity = _identity(key)
        if identity is None:
            continue
        for dtype, record in typed.items():
            if not isinstance(record, dict):
                continue
            n, shape_lambda = identity
            inputs = (record.get("config") or {}).get("inputs") or {}
            if not isinstance(inputs, dict):
                continue
            params = inputs.get("kwargs", inputs)
            if not isinstance(params, dict):
                continue
            reference_lambda = params.get("lambda", params.get("lam"))
            if isinstance(reference_lambda, dict):
                reference_lambda = reference_lambda.get("scalar")
            if reference_lambda is not None:
                if shape_lambda is not None and reference_lambda != shape_lambda:
                    continue
                shape_lambda = reference_lambda
            by_case.setdefault(((n, shape_lambda), dtype), []).append((key, record))
    for group in details:
        dtype = "torch." + group["dtype"].removeprefix("torch.")
        for point in group["result"]:
            identity = _identity(point["shape_detail"])
            matches = by_case.get((identity, dtype), [])
            if not matches:
                point["base_miss_reason"] = "operator_missing" if operator_id not in data else "shape_or_dtype_missing"
                continue
            if len(matches) != 1:
                point["base_miss_reason"] = "ambiguous_reference"
                continue
            key, record = matches[0]
            latency = record.get("latency_ms")
            if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                point["base_miss_reason"] = "invalid_baseline_latency"
                continue
            raw = latency / point["latency"]
            factor = _factor(data.get("_scaling_factors", {}), vendor, chip, record.get("bottle_neck_unit"), dtype)
            point.update(
                compared_speedup=raw / (factor or 1.0),
                base_raw_speedup=raw,
                base_data_source=str(path),
                base_shape_key=key,
                base_latency_ms=latency,
                base_scaling_factor=factor,
            )
