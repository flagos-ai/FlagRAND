"""Recompute the documented acceptance result from summary and native raw records (CPU only)."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APIS = {
    "raw32": ("RAW_BITS", "int32"),
    "uniform_f32": ("UNIFORM", "fp32"),
    "uniform_f64": ("UNIFORM", "fp64"),
    "normal_f32": ("NORMAL", "fp32"),
    "normal_f64": ("NORMAL", "fp64"),
    "lognormal_f32": ("LOGNORMAL", "fp32"),
    "lognormal_f64": ("LOGNORMAL", "fp64"),
}
BASELINES = {
    "hygon": "hygon_hiprand", "nvidia": "nvidia_curand", "iluvatar": "corex_curand",
    "metax": "mcrand", "thead": "ppu_acrand", "ascend": "cann_aclnn",
}


def shape_map(vendor):
    if vendor in {"hygon", "nvidia", "iluvatar"}:
        module = "corex" if vendor == "iluvatar" else vendor
        return importlib.import_module(f"benchmark.{module}_matrix.config").MATRIX_SHAPE_MAP
    from benchmark.muxi_matrix.config import ACCEPTANCE_DISTRIBUTION_SIZES
    return ACCEPTANCE_DISTRIBUTION_SIZES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(summary_path):
    summary = read(summary_path)
    errors = []

    def expect(condition, message):
        if not condition:
            errors.append(message)

    expect(set(summary) == {"timestamp", "total_duration", "env", "result"}, "root schema mismatch")
    names = ["PHILOX4_32_10", "RAW_BITS", "UNIFORM", "NORMAL", "LOGNORMAL"]
    expect(list(summary["result"]) == names, "delivery names/order mismatch")
    for name, result in summary["result"].items():
        expect(set(result) == {"customized", "accuracy", "performance"}, f"{name}: schema mismatch")
        accuracy = result["accuracy"]
        expect(accuracy["status"] == "Passed" and accuracy["passed"] > 0 and accuracy["failed"] == 0,
               f"{name}: accuracy {accuracy['status']}, passed={accuracy['passed']}, failed={accuracy['failed']}")
        expect(accuracy["total"] == sum(accuracy[key] for key in ("passed", "failed", "skipped")),
               f"{name}: accuracy counts disagree")
    expect(summary["result"]["PHILOX4_32_10"] == summary["result"]["RAW_BITS"] | {
        "performance": summary["result"]["RAW_BITS"]["performance"] | {"test_case": "PHILOX4_32_10"}
    }, "Philox/raw delivery aliases disagree")
    vendor = summary["env"]["flagrand"]["vendor"]
    baseline = BASELINES[vendor]
    shapes = shape_map(vendor)
    rows = []
    for api, (operator, dtype) in APIS.items():
        start = len(errors)
        expected = tuple(shapes.get(api, ()))
        row = {"api": api, "expected_shapes": len(expected), "measured_shapes": 0, "mean_speedup": None}
        rows.append(row)
        folder = summary_path.parent / api
        raw_files = list(folder.glob("performance_raw/*/records.jsonl"))
        if len(raw_files) != 1:
            errors.append(f"{api}: expected one raw matrix, found {len(raw_files)}")
            row["status"] = "incomplete"
            continue
        accuracy = read(folder / "accuracy_result.json")
        counts = Counter(item["result"] for item in accuracy.values())
        expect(counts["passed"] > 0 and set(counts) <= {"passed", "skipped"}, f"{api}: accuracy {dict(counts)}")
        perf = read(folder / "performance_result.json")[api]
        expect(perf["result"] == "passed", f"{api}: performance {perf['result']}: {perf.get('reason', '')}")
        records = [json.loads(line) for line in raw_files[0].read_text().splitlines() if line]
        manifest = read(raw_files[0].with_name("manifest.json"))
        configured = manifest.get("shape_map", manifest.get("acceptance_shape_map", {}))
        expect(bool(expected) and tuple(configured.get(api, ())) == expected, f"{api}: configured matrix mismatch")
        sides = {}
        for backend in ("flagrand_public", baseline):
            selected = [r for r in records if r["backend"] == backend and r["distribution"] == api]
            expect(Counter(r["n"] for r in selected) == Counter(expected), f"{api}/{backend}: missing/duplicate shapes")
            bad = [{"n": r["n"], "status": r["status"], "reason": r.get("reason", r.get("error"))}
                   for r in selected if r["status"] != "pass"]
            expect(not bad, f"{api}/{backend}: {bad}")
            sides[backend] = {r["n"]: r for r in selected if r["status"] == "pass"}
        data = summary["result"][operator]["performance"]["data"].get(dtype, {})
        details = data.get("details", {})
        expect(set(details) == {f"({n},)" for n in expected}, f"{api}: summary shape coverage mismatch")
        ratios = []
        for n in expected:
            if any(n not in sides[backend] for backend in sides):
                continue
            native = float(sides[baseline][n]["total_ms_median"])
            ours = float(sides["flagrand_public"][n]["total_ms_median"])
            if not all(math.isfinite(v) and v > 0 for v in (native, ours)):
                errors.append(f"{api}/{n}: invalid timing")
                continue
            ratio = native / ours
            ratios.append(ratio)
            point = details.get(f"({n},)", {})
            for key, value in (("base", native), ("gems", ours), ("speedup", ratio)):
                expect(math.isclose(float(point.get(key, math.nan)), value, rel_tol=1e-9, abs_tol=1e-12),
                       f"{api}/{n}: summary {key} disagrees with native records")
        row["measured_shapes"] = len(ratios)
        if ratios:
            mean = math.fsum(ratios) / len(ratios)
            row.update(mean_speedup=mean, min_speedup=min(ratios), max_speedup=max(ratios))
            expect(math.isclose(float(data.get("speedup", math.nan)), mean, rel_tol=1e-9),
                   f"{api}: summary mean disagrees with arithmetic mean")
            expect(mean >= 0.95, f"{api}: mean {mean:.9f} < 0.95")
        expect(len(ratios) == len(expected) and bool(ratios), f"{api}: no complete native comparison")
        row["status"] = "pass" if len(errors) == start else "fail"
    return {"summary": str(summary_path.resolve()), "vendor": vendor, "baseline": baseline,
            "minimum_api_mean": 0.95, "status": "pass" if not errors else "fail", "apis": rows, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, help="Write a separate verification record; source artifacts stay intact")
    args = parser.parse_args()
    result = check(args.summary)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in result["apis"]:
        mean = "N/A" if row["mean_speedup"] is None else f"{row['mean_speedup']:.6f}x"
        print(f"{row['api']}: {row['measured_shapes']}/{row['expected_shapes']} shapes, mean={mean}, {row['status']}")
    for error in result["errors"]:
        print(error)
    print(f"SUMMARY_JSON_PATH={result['summary']}")
    print(f"SUMMARY_JSON_CHECK={result['status'].upper()}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
