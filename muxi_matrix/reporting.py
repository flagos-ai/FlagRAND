from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


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


def print_record(record: dict[str, Any]) -> None:
    identity = (
        f"{record['backend']:<17} {record['distribution']:<15} "
        f"N={record['n']:<9}"
    )
    if record["status"] == "pass":
        speedup = record.get("speedup_vs_ppu_curand")
        speedup_text = "" if speedup is None else f" speedup_vs_curand={float(speedup):.3f}x"
        acrand_speedup = record.get("speedup_vs_ppu_acrand")
        if acrand_speedup is not None:
            speedup_text += f" speedup_vs_acrand={float(acrand_speedup):.3f}x"
        print(
            f"PASS {identity} total={record['total_ms_median']:.4f} ms "
            f"device={record.get('device_ms_median', 0.0):.4f} ms "
            f"enqueue={record['enqueue_us_median']:.2f} us"
            f"{speedup_text}"
        )
    else:
        detail = record.get("error") or record.get("validation", {}).get("checks", {})
        print(f"FAIL {identity} status={record['status']} detail={detail}")


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
        "speedup_vs_ppu_curand",
        "device_speedup_vs_ppu_curand",
        "speedup_vs_ppu_acrand",
        "device_speedup_vs_ppu_acrand",
        "library_path",
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
    lines = [
        "# Muxi Philox Validation Summary",
        "",
        f"- Preflight: `{preflight}`",
        f"- Matrix records: `{len(records)}` (`{passed}` pass, `{failed}` fail)",
        f"- Triton backend: `{compatibility.get('backend', 'unknown')}`",
        f"- Compatibility patches: `{', '.join(compatibility.get('applied', [])) or 'none'}`",
        "",
        "| Backend | Distribution | dtype | N | Status | Enqueue us | Total ms | Device ms | GE/s | Speedup vs PPU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        speedup = record.get("speedup_vs_ppu_acrand", record.get("speedup_vs_ppu_curand", 0.0))
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
