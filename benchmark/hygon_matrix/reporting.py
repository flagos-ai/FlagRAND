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
    run_directory = base / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile_name}_hygon_philox"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def print_record(record: dict[str, Any]) -> None:
    identity = f"{record['backend']:<17} {record['distribution']:<15} N={record['n']:<9}"
    if record["status"] == "pass":
        speedup = record.get("speedup_vs_hygon_hiprand")
        speedup_text = "" if speedup is None else f" speedup_vs_hiprand={float(speedup):.3f}x"
        LOGGER.info(
            "PASS %s total=%.6f ms device=%.6f ms enqueue=%.2f us%s",
            identity,
            record["total_ms_median"],
            record["device_ms_median"],
            record["enqueue_us_median"],
            speedup_text,
        )
    else:
        detail = record.get("error") or record.get("validation", {}).get("checks", {})
        LOGGER.info("FAIL %s status=%s detail=%s", identity, record["status"], detail)


def build_api_summary(records: list[dict[str, Any]], distributions: tuple[str, ...]) -> list[dict[str, Any]]:
    summary = []
    for distribution in distributions:
        points = [
            {"n": int(record["n"]), "speedup": float(record["speedup_vs_hygon_hiprand"])}
            for record in records
            if record["backend"] == "flagrand_public"
            and record["distribution"] == distribution
            and "speedup_vs_hygon_hiprand" in record
        ]
        summary.append(
            {
                "distribution": distribution,
                "points": points,
                "shape_count": len(points),
                "api_speedup": sum(point["speedup"] for point in points) / len(points) if points else None,
                "min_speedup": min((point["speedup"] for point in points), default=None),
                "max_speedup": max((point["speedup"] for point in points), default=None),
            }
        )
    return summary


def print_api_summary(summary: list[dict[str, Any]]) -> None:
    LOGGER.info("API arithmetic mean speedup: Hygon hipRAND total_ms_median / FlagRAND total_ms_median")
    for row in summary:
        if row["points"]:
            speedups = ",".join(f"{point['n']}:{point['speedup']:.3f}x" for point in row["points"])
            LOGGER.info(
                "API %-15s shapes=%d mean=%.3fx min=%.3fx max=%.3fx points=%s",
                row["distribution"],
                row["shape_count"],
                row["api_speedup"],
                row["min_speedup"],
                row["max_speedup"],
                speedups,
            )
        else:
            LOGGER.info("API %-15s mean=N/A", row["distribution"])


def write_results(run_directory: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    (run_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (run_directory / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_csv(run_directory / "records.csv", records)
    (run_directory / "summary.md").write_text(_render_summary(manifest, records), encoding="utf-8")
    write_summary_json(run_directory, manifest, records)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        "backend",
        "distribution",
        "dtype",
        "n",
        "status",
        "total_ms_median",
        "device_ms_median",
        "enqueue_us_median",
        "giga_elements_per_second",
        "effective_gb_per_second",
        "speedup_vs_hygon_hiprand",
        "device_speedup_vs_hygon_hiprand",
        "validation",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["validation"] = json.dumps(record.get("validation", {}), ensure_ascii=False)
            writer.writerow(row)


def _render_summary(manifest: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Hygon BW1000 Philox Matrix",
        "",
        "Speedup = Hygon hipRAND total time / FlagRAND total time.",
        "",
        "| Backend | API | dtype | N | Status | Total ms | Device ms | GE/s | Speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        speedup = record.get("speedup_vs_hygon_hiprand")
        lines.append(
            "| {backend} | {distribution} | {dtype} | {n} | {status} | {total:.6f} | "
            "{device:.6f} | {geps:.4f} | {speedup} |".format(
                backend=record["backend"],
                distribution=record["distribution"],
                dtype=record["dtype"],
                n=record["n"],
                status=record["status"],
                total=float(record.get("total_ms_median", 0.0)),
                device=float(record.get("device_ms_median", 0.0)),
                geps=float(record.get("giga_elements_per_second", 0.0)),
                speedup="N/A" if speedup is None else f"{float(speedup):.3f}x",
            )
        )
    lines.extend(
        (
            "",
            "## Per-API speedup",
            "",
            "| API | Shapes | Arithmetic mean | Min | Max |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for row in manifest.get("api_speedup_summary", []):
        mean = "N/A" if row["api_speedup"] is None else f"{row['api_speedup']:.3f}x"
        minimum = "N/A" if row["min_speedup"] is None else f"{row['min_speedup']:.3f}x"
        maximum = "N/A" if row["max_speedup"] is None else f"{row['max_speedup']:.3f}x"
        lines.append(f"| {row['distribution']} | {row['shape_count']} | {mean} | {minimum} | {maximum} |")
    lines.append("")
    return "\n".join(lines)
