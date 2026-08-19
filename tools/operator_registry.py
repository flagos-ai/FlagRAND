"""Load the FlagGems-compatible operator inventory."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    if value.startswith(("'", '"')):
        return ast.literal_eval(value)
    if value in {"true", "false"}:
        return value == "true"
    return value


def _load_with_stdlib(path: Path) -> dict[str, Any]:
    """Parse the deliberately small YAML subset used by the inventory."""

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    list_field: str | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#") or raw_line.strip() == "ops:":
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()

        if indent == 2 and stripped.startswith("- "):
            if current is not None:
                entries.append(current)
            current = {}
            list_field = None
            stripped = stripped[2:].strip()
            if ":" not in stripped:
                raise ValueError(f"invalid operator entry at line {line_number}: {raw_line!r}")
            key, value = (part.strip() for part in stripped.split(":", 1))
            current[key] = _scalar(value)
            continue

        if current is None:
            raise ValueError(f"operator field precedes first entry at line {line_number}")
        if indent == 4 and ":" in stripped:
            key, value = (part.strip() for part in stripped.split(":", 1))
            parsed = _scalar(value)
            current[key] = [] if parsed is None else parsed
            list_field = key if parsed is None else None
            continue
        if indent == 6 and stripped.startswith("- ") and list_field:
            item = stripped[2:].strip()
            if ":" in item:
                key, value = (part.strip() for part in item.split(":", 1))
                current[list_field].append({key: _scalar(value)})
            else:
                current[list_field].append(_scalar(item))
            continue
        raise ValueError(f"unsupported registry syntax at line {line_number}: {raw_line!r}")

    if current is not None:
        entries.append(current)
    return {"ops": entries}


def load_registry(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_with_stdlib(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data
