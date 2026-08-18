"""Ascend 910C benchmark result serialization and logging."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.summary_json import write_summary_json

LOGGER = logging.getLogger(__name__)


def create_run_directory(base: Path, profile_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = base / f"{timestamp}_{profile_name}_910c_philox"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def write_results(
    run_directory: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_directory / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_csv(run_directory / "records.csv", records)
    (run_directory / "summary.md").write_text(
        _render_summary(manifest, records),
        encoding="utf-8",
    )
    write_summary_json(run_directory, manifest, records)


def build_api_summary(
    records: list[dict[str, Any]],
    *,
    distributions: tuple[str, ...],
    shape_map: dict[str, tuple[int, ...]],
    speedup_min: float | None,
    speedup_max: float | None,
) -> dict[str, Any]:
    rows = []
    for distribution in distributions:
        points = []
        for record in records:
            if (
                record.get("backend") != "flagrand_public"
                or record.get("distribution") != distribution
                or record.get("status") != "pass"
                or "speedup_vs_cann_aclnn" not in record
            ):
                continue
            points.append(
                {
                    "n": int(record["n"]),
                    "speedup": float(record["speedup_vs_cann_aclnn"]),
                    "flagrand_total_ms": float(record["total_ms_median"]),
                    "baseline_total_ms": (float(record["total_ms_median"]) * float(record["speedup_vs_cann_aclnn"])),
                }
            )
        api_speedup = sum(point["speedup"] for point in points) / len(points) if points else None
        if api_speedup is None:
            status = "missing"
        elif speedup_min is not None and api_speedup < speedup_min:
            status = "below_min"
        elif speedup_max is not None and api_speedup > speedup_max:
            status = "above_max"
        else:
            status = "pass" if speedup_min is not None or speedup_max is not None else "reported"
        rows.append(
            {
                "distribution": distribution,
                "configured_shapes": list(shape_map[distribution]),
                "shape_count": len(points),
                "points": points,
                "api_speedup": api_speedup,
                "status": status,
            }
        )
    thresholds_enabled = speedup_min is not None or speedup_max is not None
    threshold_misses = sum(row["status"] in {"below_min", "above_max", "missing"} for row in rows)
    return {
        "baseline": "cann_aclnn",
        "formal_metric": "total_ms_median",
        "formula": "baseline total_ms_median / FlagRAND total_ms_median",
        "api_aggregation": "arithmetic mean of per-shape speedups",
        "cross_api_average": None,
        "speedup_min": speedup_min,
        "speedup_max": speedup_max,
        "thresholds_enabled": thresholds_enabled,
        "threshold_misses": threshold_misses,
        "status": "fail" if thresholds_enabled and threshold_misses else "pass",
        "rows": rows,
    }


def print_record(record: dict[str, Any]) -> None:
    identity = f"{record['backend']:<17} {record['distribution']:<15} " f"N={record['n']:<9}"
    if record["status"] == "pass":
        device = record.get("device_ms_median")
        device_text = "n/a" if device is None else f"{float(device):.4f} ms"
        speedup = record.get("speedup_vs_cann_aclnn")
        speedup_text = "" if speedup is None else f" speedup={float(speedup):.3f}x"
        LOGGER.info(
            f"PASS {identity} total={record['total_ms_median']:.4f} ms "
            f"device={device_text} enqueue={record['enqueue_us_median']:.2f} us"
            f"{speedup_text}"
        )
    else:
        detail = record.get("error") or record.get("validation", {}).get("checks", {})
        LOGGER.info("FAIL %s status=%s detail=%s", identity, record["status"], detail)


def print_api_summary(summary: dict[str, Any]) -> None:
    LOGGER.info("Per-API speedup summary baseline=cann_aclnn metric=total_ms_median " "cross_api_average=disabled")
    for row in summary["rows"]:
        if row["points"]:
            point_text = ",".join(f"N={point['n']}:{point['speedup']:.3f}x" for point in row["points"])
            api_text = f"{row['api_speedup']:.3f}x"
        else:
            point_text = "n/a"
            api_text = "n/a"
        LOGGER.info(f"API {row['status']:<10} {row['distribution']:<15} " f"shapes=[{point_text}] mean={api_text}")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    preferred = [
        "backend",
        "generator",
        "implementation_path",
        "distribution",
        "dtype",
        "n",
        "seed",
        "status",
        "timing_calls_per_repeat",
        "enqueue_us_median",
        "total_ms_median",
        "total_ms_min",
        "total_ms_max",
        "device_ms_median",
        "device_ms_min",
        "device_ms_max",
        "giga_elements_per_second",
        "effective_gb_per_second",
        "speedup_vs_cann_aclnn",
        "device_speedup_vs_cann_aclnn",
        "validation",
        "components",
        "error_type",
        "error",
    ]
    extra = sorted({key for record in records for key in record} - set(preferred))
    columns = preferred + extra
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {
                key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value)
                for key, value in record.items()
            }
            writer.writerow(row)


def _render_summary(manifest: dict[str, Any], records: list[dict[str, Any]]) -> str:
    passed = sum(record.get("status") == "pass" for record in records)
    failed = len(records) - passed
    environment = manifest.get("environment", {})
    versions = environment.get("versions", {})
    lines = [
        "# Huawei Ascend 910C FlagRAND Validation Summary",
        "",
        f"- Preflight: `{manifest.get('preflight', {}).get('status', 'not_run')}`",
        f"- Matrix: `{len(records)}` records (`{passed}` pass, `{failed}` fail)",
        f"- Device: `{environment.get('device_name', 'unknown')}`",
        f"- CANN: `{versions.get('cann', 'unknown')}`",
        f"- Triton backend: `{environment.get('triton_target', {}).get('backend', 'unknown')}`",
        "- Formal metric: `total_ms_median` (end-to-end wall time)",
        "- Device event: secondary diagnostic only",
        "",
        "| Backend | API | dtype | N | Status | Total ms | Device ms | GE/s | Speedup vs ACLNN | Path |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {backend} | {distribution} | {dtype} | {n} | {status} | "
            "{total:.5f} | {device:.5f} | {geps:.4f} | {speedup:.4f} | {path} |".format(
                backend=record["backend"],
                distribution=record["distribution"],
                dtype=record["dtype"],
                n=record["n"],
                status=record["status"],
                total=float(record.get("total_ms_median", 0.0)),
                device=float(record.get("device_ms_median", 0.0)),
                geps=float(record.get("giga_elements_per_second", 0.0)),
                speedup=float(record.get("speedup_vs_cann_aclnn", 0.0)),
                path=record.get("implementation_path", "n/a"),
            )
        )
    api_summary = manifest.get("api_speedup_summary")
    if api_summary:
        lines.extend(("", "## Per-API speedup", ""))
        lines.append("Cross-API averaging is intentionally not reported.")
        lines.extend(
            (
                "",
                "| API | Shapes | Arithmetic mean | Status |",
                "|---|---:|---:|---:|",
            )
        )
        for row in api_summary["rows"]:
            shapes = ", ".join(str(point["n"]) for point in row["points"]) or "n/a"
            speedup = "n/a" if row["api_speedup"] is None else f"{row['api_speedup']:.4f}x"
            lines.append(f"| {row['distribution']} | {shapes} | {speedup} | {row['status']} |")
    failures = [record for record in records if record.get("status") != "pass"]
    if failures:
        lines.extend(("", "## Failures", ""))
        for record in failures:
            detail = record.get("error") or record.get("validation", {}).get("checks", {})
            lines.append(
                f"- `{record['backend']}/{record['distribution']}/N={record['n']}`: "
                f"`{record['status']}` — `{detail}`"
            )
    lines.append("")
    return "\n".join(lines)
