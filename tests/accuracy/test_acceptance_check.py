"""CPU-only checks for incomplete, misleading and below-target delivery artifacts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.check_acceptance import APIS, check, shape_map


class AcceptanceCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.summary = self.root / "summary.json"
        results = {}
        for api, (operator, dtype) in APIS.items():
            folder = self.root / api
            raw = folder / "performance_raw" / "run"
            raw.mkdir(parents=True)
            shapes = shape_map("hygon")[api]
            self.write(raw / "manifest.json", {"shape_map": {api: shapes}})
            self.write(folder / "accuracy_result.json", {api: {"result": "passed"}})
            self.write(folder / "performance_result.json", {api: {"result": "passed"}})
            records = [{"backend": backend, "distribution": api, "n": n, "status": "pass", "total_ms_median": 1.0}
                       for n in shapes for backend in ("flagrand_public", "hygon_hiprand")]
            (raw / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            entry = results.setdefault(operator, {
                "customized": True,
                "accuracy": {"total": 0, "passed": 0, "skipped": 0, "failed": 0, "status": "Passed", "details": {}},
                "performance": {"status": "Passed", "test_case": operator, "data": {}},
            })
            entry["accuracy"]["total"] += 1
            entry["accuracy"]["passed"] += 1
            entry["performance"]["data"][dtype] = {
                "result": "OK", "speedup": 1.0,
                "details": {f"({n},)": {"base": 1.0, "gems": 1.0, "speedup": 1.0} for n in shapes},
            }
        philox = copy.deepcopy(results["RAW_BITS"])
        philox["performance"]["test_case"] = "PHILOX4_32_10"
        self.payload = {"timestamp": "2026-09-14 00:00:00", "total_duration": "0:01:00",
                        "env": {"flagrand": {"vendor": "hygon"}}, "result": {"PHILOX4_32_10": philox, **results}}
        self.write(self.summary, self.payload)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def test_complete_artifacts_pass(self):
        self.assertEqual(check(self.summary)["status"], "pass")

    def test_one_slow_shape_is_allowed(self):
        path = self.root / "normal_f32/performance_raw/run/records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[1]["total_ms_median"] = 0.5
        path.write_text("\n".join(json.dumps(r) for r in records))
        data = self.payload["result"]["NORMAL"]["performance"]["data"]["fp32"]
        data["details"]["(2,)"].update(base=0.5, speedup=0.5)
        data["speedup"] = 1 - 0.5 / 42
        self.write(self.summary, self.payload)
        self.assertEqual(check(self.summary)["status"], "pass")

    def test_missing_shape_cannot_pass_by_averaging_survivors(self):
        path = self.root / "raw32/performance_raw/run/records.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[2:]))
        self.assertEqual(check(self.summary)["status"], "fail")

    def test_forged_summary_mean_is_rejected(self):
        self.payload["result"]["NORMAL"]["performance"]["data"]["fp64"]["speedup"] = 8.0
        self.write(self.summary, self.payload)
        self.assertIn("normal_f64: summary mean", "\n".join(check(self.summary)["errors"]))

    def test_slow_api_is_not_hidden_by_other_apis(self):
        path = self.root / "lognormal_f64/performance_raw/run/records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if record["backend"] == "hygon_hiprand":
                record["total_ms_median"] = 0.9
        path.write_text("\n".join(json.dumps(r) for r in records))
        data = self.payload["result"]["LOGNORMAL"]["performance"]["data"]["fp64"]
        data["speedup"] = 0.9
        for point in data["details"].values():
            point.update(base=0.9, speedup=0.9)
        self.write(self.summary, self.payload)
        self.assertIn("lognormal_f64: mean 0.900000000 < 0.95", check(self.summary)["errors"])

    def test_unsupported_baseline_is_not_a_pass(self):
        path = self.root / "uniform_f64/performance_raw/run/records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[1]["status"] = "unsupported"
        path.write_text("\n".join(json.dumps(r) for r in records))
        self.assertEqual(check(self.summary)["status"], "fail")

    def test_empty_accuracy_is_not_a_pass(self):
        self.write(self.root / "normal_f64/accuracy_result.json", {})
        self.assertEqual(check(self.summary)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
