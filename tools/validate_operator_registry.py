"""Validate that registered operators have public APIs and test coverage."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

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
}


def _parse_inline_list(value: str) -> list[str]:
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"expected an inline YAML list, got {value!r}")
    return [item.strip() for item in value[1:-1].split(",") if item.strip()]


def _load_with_stdlib(registry_path: Path) -> dict[str, Any]:
    """Parse the deliberately flat operator registry without third-party packages."""

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "ops:":
            continue
        if stripped.startswith("- "):
            if current is not None:
                entries.append(current)
            current = {}
            stripped = stripped[2:].strip()
        if current is None or ":" not in stripped:
            raise ValueError(f"unsupported registry syntax at line {line_number}: {raw_line!r}")
        key, value = (part.strip() for part in stripped.split(":", 1))
        current[key] = _parse_inline_list(value) if value.startswith("[") else value
    if current is not None:
        entries.append(current)
    return {"ops": entries}


def _load_registry(registry_path: Path) -> dict[str, Any]:
    if os.environ.get("FLAGRAND_FORCE_STDLIB_YAML") != "1":
        try:
            import yaml

            loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except ModuleNotFoundError:
            pass
    return _load_with_stdlib(registry_path)


def validate_registry(project_root: Path) -> list[str]:
    registry_path = project_root / "conf" / "operators.yaml"
    data = _load_registry(registry_path)
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
        if entry["for"] != f"flagrand.ops.philox.{operator_id}":
            errors.append(f"{operator_id}: public API path does not match its id")
        if f'"{operator_id}"' not in facade:
            errors.append(f"{operator_id}: absent from flagrand.ops.philox.__all__")
        if not {"accuracy", "performance"}.issubset(set(entry["stages"])):
            errors.append(f"{operator_id}: stages must contain accuracy and performance")
        for field in ("accuracy_test", "performance_test"):
            test_path = project_root / str(entry[field])
            if not test_path.is_file():
                errors.append(f"{operator_id}: missing {field} file {entry[field]}")
                continue
            test_source = test_path.read_text(encoding="utf-8")
            if f"pytest.mark.{operator_id}" not in test_source:
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
                if f"pytest.mark.{operator_id}" not in test_source:
                    errors.append(f"{operator_id}: {test_name} lacks pytest.mark.{operator_id}")

    if len(ids) != len(set(ids)):
        errors.append("operator ids must be unique")
    if set(ids) != EXPECTED_OPERATORS:
        errors.append("operator registry must contain exactly the seven public APIs; " f"found={sorted(set(ids))}")
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
    LOGGER.info("Operator registry: PASS (7/7 APIs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
