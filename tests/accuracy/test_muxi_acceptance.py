from __future__ import annotations

import json
import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import Mock, patch

from benchmark.muxi_matrix.config import parse_args
from benchmark.muxi_matrix.reporting import (
    _render_acceptance_summary,
    build_acceptance_summary,
    format_acceptance_target,
    print_acceptance_summary,
)
from benchmark.summary_json import aggregate_delivery_results
from tools.run_tests import RunnerConfig, run_performance


def _records(distribution: str, speedups: tuple[float, ...]) -> list[dict]:
    return [
        {
            "backend": "flagrand_public",
            "distribution": distribution,
            "dtype": "int32",
            "n": 1025 + index,
            "status": "pass",
            "total_ms_median": 0.02,
            "speedup_vs_mcrand": speedup,
        }
        for index, speedup in enumerate(speedups)
    ]


def _summary(speedups: tuple[float, ...], **kwargs) -> dict:
    records = _records("raw32", speedups)
    return build_acceptance_summary(
        records,
        distributions=("raw32",),
        shape_map={"raw32": tuple(record["n"] for record in records)},
        baseline="mcrand",
        min_speedup=0.95,
        **kwargs,
    )


class TestMuxiAcceptance(unittest.TestCase):
    def test_metax_defaults_to_per_api_minimum_only(self) -> None:
        with patch.object(sys, "argv", ["benchmark.run_muxi", "--acceptance-profile", "--strict-acceptance"]):
            args = parse_args(Path.cwd())
        self.assertEqual(args.acceptance_min, 0.95)
        self.assertIsNone(args.acceptance_max)
        self.assertEqual(format_acceptance_target(args.acceptance_min, args.acceptance_max), ">=0.95")

    def test_only_api_mean_is_thresholded(self) -> None:
        for speedups, expected in (
            ((0.94,), "warn"),
            ((0.95,), "pass"),
            ((1.50,), "pass"),
            ((1.580,), "pass"),
            ((1.536,), "pass"),
            ((5.0,), "pass"),
            ((0.5, 1.5), "pass"),
        ):
            with self.subTest(speedups=speedups):
                summary = _summary(speedups)
                self.assertEqual(summary["status"], expected)
                self.assertEqual(summary["rows"][0]["status"], expected)
                self.assertEqual(summary["rows"][0]["api_speedup"], sum(speedups) / len(speedups))

    def test_api_results_are_not_averaged_together(self) -> None:
        summary = build_acceptance_summary(
            _records("raw32", (2.0,)) + _records("uniform_f32", (0.9,)),
            distributions=("raw32", "uniform_f32"),
            shape_map={"raw32": (1025,), "uniform_f32": (1025,)},
            baseline="mcrand",
            min_speedup=0.95,
        )
        self.assertEqual(summary["status"], "warn")
        self.assertEqual(summary["pass_count"], 1)
        self.assertIsNone(summary["cross_api_average"])

    def test_high_speedup_has_no_special_log_or_markdown_notice(self) -> None:
        summary = _summary((1.580,))
        with self.assertLogs("benchmark.muxi_matrix.reporting", level="INFO") as logs:
            print_acceptance_summary(summary)
        output = "\n".join(logs.output)
        self.assertIn("target=>=0.95", output)
        self.assertIn("Acceptance status: pass (1/1 APIs meeting target)", output)
        markdown = "\n".join(_render_acceptance_summary(summary))
        self.assertIn("`>=0.95`", markdown)
        for text in (output, markdown):
            self.assertNotIn("1.50", text)
            self.assertNotIn("warn", text.lower())
        self.assertIsNone(json.loads(json.dumps(summary, allow_nan=False))["max_speedup"])

    def test_ppu_retains_its_configured_upper_bound(self) -> None:
        muxi_module = ModuleType("benchmark.run_muxi")
        muxi_module.main = Mock(return_value=0)
        with patch.dict(sys.modules, {"benchmark.run_muxi": muxi_module}):
            with self.assertRaises(SystemExit) as result:
                runpy.run_module("benchmark.run_ppu", run_name="__main__")
        self.assertEqual(result.exception.code, 0)
        defaults = muxi_module.main.call_args.kwargs
        with patch.object(sys, "argv", ["benchmark.run_ppu", "--acceptance-profile"]):
            args = parse_args(Path.cwd(), **defaults)
        self.assertEqual(args.acceptance_max, 1.50)
        self.assertEqual(_summary((1.580,), max_speedup=args.acceptance_max)["status"], "warn")
        self.assertEqual(format_acceptance_target(args.acceptance_min, args.acceptance_max), "[0.95, 1.50]")

    def test_unified_summary_keeps_high_speedup_passed(self) -> None:
        with TemporaryDirectory() as directory:
            config = RunnerConfig(Path(directory), "metax", "acceptance", False, 60)
            exit_code = 0 if _summary((1.580,))["status"] == "pass" else 1
            with patch("tools.run_tests.run_cmd", return_value=exit_code):
                with patch("tools.run_tests._read_benchmark_records", return_value=_records("raw32", (1.580,))):
                    performance = run_performance(0, {"id": "raw32"}, config)
        self.assertEqual(performance["status"], "Passed")
        results = aggregate_delivery_results(
            {"raw32": {"accuracy": {"status": "Passed", "total": 1, "passed": 1}, "performance": performance}}
        )
        for name in ("PHILOX4_32_10", "RAW_BITS"):
            self.assertEqual(results[name]["performance"]["status"], "Passed")
            self.assertEqual(results[name]["performance"]["data"]["int32"]["speedup"], 1.580)


if __name__ == "__main__":
    unittest.main()
