"""Muxi benchmark result serialization and logging."""

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
    run_directory = base / f"{timestamp}_{profile_name}_philox"
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


def build_acceptance_summary(
    records: list[dict[str, Any]],
    *,
    distributions: tuple[str, ...],
    shape_map: dict[str, tuple[int, ...]],
    baseline: str,
    min_speedup: float,
    max_speedup: float | None = None,
) -> dict[str, Any]:
    speedup_key = f"speedup_vs_{baseline}"
    rows = []
    for distribution in distributions:
        distribution_records = [
            record
            for record in records
            if record.get("backend") == "flagrand_public"
            and record.get("distribution") == distribution
            and record.get("status") == "pass"
        ]
        points = []
        for record in distribution_records:
            speedup = record.get(speedup_key)
            if speedup is None:
                continue
            points.append(
                {
                    "n": int(record["n"]),
                    "speedup": float(speedup),
                    "total_ms": float(record["total_ms_median"]),
                    "baseline_total_ms": (float(record["total_ms_median"]) * float(speedup)),
                }
            )
        if points:
            api_speedup = sum(point["speedup"] for point in points) / len(points)
            meets_target = api_speedup >= min_speedup and (max_speedup is None or api_speedup <= max_speedup)
            status = "pass" if meets_target else "warn"
        else:
            api_speedup = None
            status = "missing"
        rows.append(
            {
                "distribution": distribution,
                "configured_shapes": list(shape_map.get(distribution, ())),
                "shape_count": len(points),
                "points": points,
                "api_speedup": api_speedup,
                "status": status,
            }
        )
    pass_count = sum(row["status"] == "pass" for row in rows)
    return {
        "baseline": baseline,
        "speedup_key": speedup_key,
        "min_speedup": float(min_speedup),
        "max_speedup": float(max_speedup) if max_speedup is not None else None,
        "api_count": len(rows),
        "pass_count": pass_count,
        "status": "pass" if pass_count == len(rows) else "warn",
        "cross_api_average": None,
        "rows": rows,
    }


def format_acceptance_target(min_speedup: float, max_speedup: float | None) -> str:
    if max_speedup is None:
        return f">={min_speedup:.2f}"
    return f"[{min_speedup:.2f}, {max_speedup:.2f}]"


def print_acceptance_summary(summary: dict[str, Any]) -> None:
    LOGGER.info(
        "Acceptance speedup summary "
        f"baseline={summary['baseline']} "
        f"target={format_acceptance_target(summary['min_speedup'], summary['max_speedup'])} "
        "cross_api_average=disabled"
    )
    for row in summary["rows"]:
        points = row["points"]
        if points:
            shapes = ",".join(str(point["n"]) for point in points)
            shape_speedups = ",".join(f"{point['speedup']:.3f}x" for point in points)
            api_speedup = f"{row['api_speedup']:.3f}x"
        else:
            shapes = ",".join(str(size) for size in row["configured_shapes"]) or "n/a"
            shape_speedups = "n/a"
            api_speedup = "n/a"
        LOGGER.info(
            f"ACCEPTANCE {row['status']:<7} {row['distribution']:<15} "
            f"shapes={shapes:<20} shape_speedups={shape_speedups:<16} "
            f"api_speedup={api_speedup}"
        )
    LOGGER.info(
        f"Acceptance status: {summary['status']} "
        f"({summary['pass_count']}/{summary['api_count']} APIs meeting target)"
    )


def print_record(record: dict[str, Any]) -> None:
    identity = f"{record['backend']:<17} {record['distribution']:<15} " f"N={record['n']:<9}"
    if record["status"] == "pass":
        speedup_text = ""
        for baseline in ("mcrand", "ppu_curand", "ppu_acrand"):
            speedup = record.get(f"speedup_vs_{baseline}")
            if speedup is not None:
                speedup_text += f" speedup_vs_{baseline}={float(speedup):.3f}x"
        LOGGER.info(
            f"PASS {identity} total={record['total_ms_median']:.4f} ms "
            f"device={record.get('device_ms_median', 0.0):.4f} ms "
            f"enqueue={record['enqueue_us_median']:.2f} us"
            f"{speedup_text}"
        )
    else:
        detail = record.get("error") or record.get("validation", {}).get("checks", {})
        LOGGER.info("FAIL %s status=%s detail=%s", identity, record["status"], detail)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        "backend",
        "generator",
        "distribution",
        "dtype",
        "n",
        "seed",
        "status",
        "enqueue_us_median",
        "total_ms_median",
        "total_ms_min",
        "total_ms_max",
        "device_ms_median",
        "device_ms_min",
        "device_ms_max",
        "giga_elements_per_second",
        "effective_gb_per_second",
        "speedup_vs_mcrand",
        "device_speedup_vs_mcrand",
        "speedup_vs_ppu_curand",
        "device_speedup_vs_ppu_curand",
        "speedup_vs_ppu_acrand",
        "device_speedup_vs_ppu_acrand",
        "library_path",
        "library_version",
        "libtuner",
        "validation",
        "error_type",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["validation"] = json.dumps(record.get("validation", {}), ensure_ascii=False)
            row["libtuner"] = json.dumps(record.get("libtuner", {}), ensure_ascii=False)
            writer.writerow(row)


def _render_summary(manifest: dict[str, Any], records: list[dict[str, Any]]) -> str:
    passed = sum(record["status"] == "pass" for record in records)
    failed = len(records) - passed
    preflight = manifest.get("preflight", {}).get("status", "error")
    compatibility = manifest.get("compatibility", {})
    baseline = str(manifest.get("acceptance_baseline", "mcrand"))
    speedup_key = f"speedup_vs_{baseline}"
    lines = [
        "# Muxi Philox Validation Summary",
        "",
        f"- Preflight: `{preflight}`",
        f"- Matrix records: `{len(records)}` (`{passed}` pass, `{failed}` fail)",
        f"- Triton backend: `{compatibility.get('backend', 'unknown')}`",
        f"- Compatibility patches: `{', '.join(compatibility.get('applied', [])) or 'none'}`",
        "",
        "| Backend | Distribution | dtype | N | Status | Enqueue us | "
        f"Total ms | Device ms | GE/s | Speedup vs {baseline} |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        speedup = record.get(speedup_key, 0.0)
        lines.append(
            "| {backend} | {distribution} | {dtype} | {n} | {status} | "
            "{enqueue:.3f} | {total:.5f} | {device:.5f} | {geps:.4f} | {speedup:.4f} |".format(
                backend=record["backend"],
                distribution=record["distribution"],
                dtype=record["dtype"],
                n=record["n"],
                status=record["status"],
                enqueue=float(record.get("enqueue_us_median", 0.0)),
                total=float(record.get("total_ms_median", 0.0)),
                device=float(record.get("device_ms_median", 0.0)),
                geps=float(record.get("giga_elements_per_second", 0.0)),
                speedup=float(speedup),
            )
        )
    acceptance_summary = manifest.get("acceptance_summary")
    if acceptance_summary:
        lines.extend(_render_acceptance_summary(acceptance_summary))
    failures = [record for record in records if record["status"] != "pass"]
    if failures:
        lines.extend(("", "## Failures", ""))
        for record in failures:
            detail = record.get("error") or record.get("validation", {}).get("checks", {})
            lines.append(
                f"- `{record['backend']}/{record['distribution']}/N={record['n']}`: "
                f"`{record['status']}` - `{detail}`"
            )
    lines.append("")
    return "\n".join(lines)


def _render_acceptance_summary(summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Acceptance Speedup Summary",
        "",
        f"- Baseline: `{summary['baseline']}`",
        f"- Per-API mean speedup target: `{format_acceptance_target(summary['min_speedup'], summary['max_speedup'])}`",
        "- Cross-API average: `not reported`",
        (
            f"- Status: `{summary['status']}` "
            f"(`{summary['pass_count']}`/`{summary['api_count']}` APIs meeting target)"
        ),
        "",
        "| API | Shapes | Shape speedups | API speedup | Status |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        if row["points"]:
            shapes = ", ".join(str(point["n"]) for point in row["points"])
            shape_speedups = ", ".join(f"{point['speedup']:.3f}x" for point in row["points"])
            api_speedup = f"{row['api_speedup']:.3f}x"
        else:
            shapes = ", ".join(str(size) for size in row["configured_shapes"]) or "n/a"
            shape_speedups = "n/a"
            api_speedup = "n/a"
        lines.append(f"| {row['distribution']} | {shapes} | {shape_speedups} | " f"{api_speedup} | {row['status']} |")
    return lines
