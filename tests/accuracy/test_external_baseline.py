"""Reference-file comparisons used by flagos-optest's performance reports."""

import json

import pytest

from benchmark.external_baseline import apply_external_baseline


def test_external_latency_keeps_native_ratio_and_records_scaling(tmp_path):
    source = tmp_path / "baseline.json"
    source.write_text(
        json.dumps(
            {
                "uniform_f32": {
                    "shapes": {"[1024]": {"torch.float32": {"latency_ms": 4.0, "bottle_neck_unit": "mem"}}}
                },
                "_scaling_factors": {
                    "chips": [{"vendor": "iluvatar", "chip": "BI-V150", "measured": {"bandwidth_gbps": 2.0}}]
                },
            }
        )
    )
    points = [
        {"shape_detail": "(1024,)", "latency": 1.0, "latency_base": 1.5, "speedup": 1.5},
        {"shape_detail": "(2048,)", "latency": 2.0, "latency_base": 3.0, "speedup": 1.5},
    ]
    apply_external_baseline(
        "uniform_f32", [{"dtype": "torch.float32", "result": points}], source, "iluvatar", "BI-V150"
    )
    assert points[0]["speedup"] == 1.5
    assert points[0]["base_raw_speedup"] == 4.0
    assert points[0]["compared_speedup"] == 2.0
    assert points[0]["base_shape_key"] == "[1024]"
    assert points[1]["base_miss_reason"]
    assert "compared_speedup" not in points[1]


def test_poisson_matches_the_recorded_lambda_and_dtype(tmp_path):
    source = tmp_path / "baseline.json"
    source.write_text(
        json.dumps(
            {
                "philox_poisson": {
                    "shapes": {
                        "[1024]": {"torch.int32": {"latency_ms": 20}},
                        "(1024,);lambda=10": {"torch.int32": {"latency_ms": 3}},
                        "(1024,);lambda=30": {"torch.float32": {"latency_ms": 7}},
                    }
                }
            }
        )
    )
    points = [{"shape_detail": f"(1024,);lambda={lam}", "latency": 1.0, "speedup": 1.5} for lam in (1, 10, 30)]
    apply_external_baseline(
        "philox_poisson", [{"dtype": "torch.int32", "result": points}], source, "iluvatar", "BI-V150"
    )
    assert "compared_speedup" not in points[0]
    assert points[1]["compared_speedup"] == 3.0
    assert "compared_speedup" not in points[2]


def test_missing_reference_file_keeps_native_speedup(tmp_path):
    row = {"shape_detail": "(32,)", "latency": 2.0, "latency_base": 3.0, "speedup": 1.5}
    with pytest.warns(UserWarning, match="Cannot read --base-data"):
        apply_external_baseline(
            "uniform_f32",
            [{"dtype": "torch.float32", "result": [row]}],
            tmp_path / "absent.json",
            "iluvatar",
            "BI-V150",
        )
    assert row["speedup"] == 1.5
    assert row["base_miss_reason"] == "base_data_unavailable"
    assert "compared_speedup" not in row
