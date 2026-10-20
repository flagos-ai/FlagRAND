"""Validate that registered operators have public APIs and test coverage."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.rng_delivery.config import OPERATORS, split_operator  # noqa: E402
from tools.operator_registry import load_registry  # noqa: E402

LOGGER = logging.getLogger(__name__)
REQUIRED_FIELDS = {
    "id",
    "description",
    "for",
    "labels",
    "kind",
    "dtypes",
    "accuracy_test",
    "platform_accuracy_tests",
    "performance_test",
    "stages",
}
EXPECTED_OPERATORS = {
    "raw32",
    "uniform_f32",
    "uniform_f64",
    "normal_f32",
    "normal_f64",
    "lognormal_f32",
    "lognormal_f64",
} | set(OPERATORS)

FAMILY_METHODS = {
    "raw32": "generate",
    "uniform_f32": "generate_uniform",
    "uniform_f64": "generate_uniform_double",
    "normal_f32": "generate_normal",
    "normal_f64": "generate_normal_double",
    "lognormal_f32": "generate_lognormal",
    "lognormal_f64": "generate_lognormal_double",
    "poisson": "generate_poisson",
}


def validate_registry(project_root: Path) -> list[str]:
    registry_path = project_root / "conf" / "operators.yaml"
    data = load_registry(registry_path)
    entries = data.get("ops") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return ["conf/operators.yaml must contain an 'ops' list"]

    errors: list[str] = []
    ids: list[str] = []
    facade = (project_root / "src" / "flagrand" / "ops" / "philox.py").read_text(encoding="utf-8")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"ops[{index}] must be a mapping")
            continue
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            errors.append(f"ops[{index}] is missing fields: {sorted(missing)}")
            continue
        operator_id = str(entry["id"])
        ids.append(operator_id)
        family = operator_id in OPERATORS
        expected_path = (
            f"flagrand.curand.{FAMILY_METHODS[split_operator(operator_id)[1]]}"
            if family
            else f"flagrand.ops.philox.{operator_id}"
        )
        if entry["for"] != [expected_path]:
            errors.append(f"{operator_id}: public API path does not match its id")
        if not family and f'"{operator_id}"' not in facade:
            errors.append(f"{operator_id}: absent from flagrand.ops.philox.__all__")
        stages = entry["stages"]
        if not isinstance(stages, list) or not stages or not all(isinstance(stage, dict) for stage in stages):
            errors.append(f"{operator_id}: stages must use the FlagGems maturity format")
        elif next(reversed(stages[-1])) not in {"alpha", "beta", "stable", "removed"}:
            errors.append(f"{operator_id}: unsupported maturity stage")
        for field in ("accuracy_test", "performance_test"):
            test_path = project_root / str(entry[field])
            if not test_path.is_file():
                errors.append(f"{operator_id}: missing {field} file {entry[field]}")
                continue
            test_source = test_path.read_text(encoding="utf-8")
            if not family and f"pytest.mark.{operator_id}" not in test_source:
                errors.append(f"{operator_id}: {entry[field]} lacks pytest.mark.{operator_id}")
        platform_accuracy_tests = entry["platform_accuracy_tests"]
        if not isinstance(platform_accuracy_tests, list) or not platform_accuracy_tests:
            errors.append(f"{operator_id}: platform_accuracy_tests must be a non-empty list")
        else:
            for test_name in platform_accuracy_tests:
                test_path = project_root / str(test_name)
                if not test_path.is_file():
                    errors.append(f"{operator_id}: missing platform accuracy file {test_name}")
                    continue
                test_source = test_path.read_text(encoding="utf-8")
                if not family and f"pytest.mark.{operator_id}" not in test_source:
                    errors.append(f"{operator_id}: {test_name} lacks pytest.mark.{operator_id}")

    if len(ids) != len(set(ids)):
        errors.append("operator ids must be unique")
    if set(ids) != EXPECTED_OPERATORS:
        errors.append(f"operator registry does not match the public API inventory; found={sorted(set(ids))}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors = validate_registry(args.project_root.resolve())
    if errors:
        for error in errors:
            LOGGER.error("REGISTRY ERROR: %s", error)
        return 1
    LOGGER.info("Operator registry: PASS (%d APIs)", len(EXPECTED_OPERATORS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
