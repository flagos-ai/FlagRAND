"""Write native records and the existing FlagOS summary hierarchy."""

import csv
import json
from datetime import datetime

from benchmark.summary_json import (
    ARITHMETIC_TARGET_PLATFORMS,
    _accuracy_result,
    aggregate_delivery_results,
    collect_environment,
)


def write_results(directory, manifest, records):
    from tools.run_tests import normalise_performance, parse_performance_data

    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    columns = [
        "backend",
        "generator",
        "distribution",
        "api",
        "dtype",
        "n",
        "lambda",
        "status",
        "total_ms_median",
        "device_ms_median",
        "enqueue_us_median",
    ]
    baseline_names = set(manifest.get("acceptance_baselines", {}).values()) or {manifest["acceptance_baseline"]}
    columns.extend(f"speedup_vs_{name}" for name in sorted(baseline_names))
    with (directory / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    api_results = {}
    for operator in manifest["distributions"]:
        path = directory / f"{operator}.performance.json"
        path.write_text(json.dumps(normalise_performance(operator, records), indent=2) + "\n")
        api_results[operator] = {
            "customized": True,
            "accuracy": _accuracy_result(
                [row for row in records if row["backend"] == "flagrand_public" and row["distribution"] == operator]
            ),
            "performance": parse_performance_data(operator, path),
        }
        expected = len(manifest["shape_map"][operator]) * len(manifest["lambda_map"][operator])
        performance = api_results[operator]["performance"]
        observed = sum(len(item["details"]) for item in performance["data"].values())
        performance.update(
            profile=manifest["profile"],
            expected_point_count=expected,
            observed_point_count=observed,
            complete_shape_set=observed == expected and performance["status"] == "Passed",
        )
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": collect_environment(manifest),
        "result": aggregate_delivery_results(api_results, platform=manifest["platform"]),
        "profile": manifest["profile"],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    arithmetic = manifest["platform"] in ARITHMETIC_TARGET_PLATFORMS
    lines = [
        f"# {manifest['platform']} RNG results",
        "",
        f"Primary timing: total_ms_median; speedup={manifest['acceptance_baseline']}/FlagRAND.",
        (
            "Arithmetic mean of per-shape/lambda speedups; each API has equal weight in the family arithmetic mean."
            if arithmetic
            else "The existing speedup field is an arithmetic mean; geometric_mean is reported separately."
        ),
        "",
        (
            "| API | dtype | Points | Arithmetic mean | Status |"
            if arithmetic
            else "| API | dtype | Points | Arithmetic mean | Geometric mean | Status |"
        ),
        "|---|---|---:|---:|---|" if arithmetic else "|---|---|---:|---:|---:|---|",
    ]
    for operator, result in api_results.items():
        for dtype, data in result["performance"]["data"].items():
            if arithmetic:
                lines.append(
                    f"| {operator} | {dtype} | {len(data['details'])} | {data['speedup']:.6f} | "
                    f"{result['performance']['status']} |"
                )
                continue
            geometric = data.get("geometric_mean")
            geometric_text = f"{geometric:.6f}" if geometric is not None else "N/A"
            lines.append(
                f"| {operator} | {dtype} | {len(data['details'])} | {data['speedup']:.6f} | "
                f"{geometric_text} | {result['performance']['status']} |"
            )
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
