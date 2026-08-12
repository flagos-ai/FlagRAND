"""Ascend 910C runtime and CANN baseline compatibility checks."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

REQUIRED_ACLNN_SYMBOLS = (
    "aclnnInplaceRandomGetWorkspaceSize",
    "aclnnInplaceRandom",
    "aclnnInplaceUniformGetWorkspaceSize",
    "aclnnInplaceUniform",
    "aclnnInplaceNormalGetWorkspaceSize",
    "aclnnInplaceNormal",
)


def import_torch_npu() -> Any:
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is unavailable. Run inside the requested CANN 9.0 container "
            "and activate its torch_npu environment."
        ) from exc
    return torch_npu


def select_device(device_index: int) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false; no Ascend NPU is visible.")
    torch.npu.set_device(device_index)
    probe = torch.empty(1, device=f"npu:{device_index}", dtype=torch.float32)
    probe.fill_(1.0)
    torch.npu.synchronize()


def collect_and_validate_environment(
    device_index: int,
    *,
    allow_non_cann9: bool,
    allow_unverified_910c: bool,
    require_baseline: bool,
) -> dict[str, Any]:
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    target_fields = {name: _json_value(getattr(target, name, None)) for name in ("backend", "arch", "warp_size")}
    device_name = _device_name(device_index)
    device_properties = _device_properties(device_index)
    smi = _run_optional(("npu-smi", "info"), timeout=15)
    smi_board = _run_optional(
        ("npu-smi", "info", "-t", "board", "-i", str(device_index), "-c", "0"),
        timeout=15,
    )
    cann_root = _resolve_cann_root()
    cann_version = _read_cann_version(cann_root)
    opapi_path = _find_libopapi(cann_root)
    symbols = _probe_symbols(opapi_path, REQUIRED_ACLNN_SYMBOLS)

    evidence_text = " ".join(
        (
            device_name,
            str(device_properties),
            str(target_fields),
            smi.get("stdout", ""),
            smi_board.get("stdout", ""),
        )
    ).lower()
    evidence_markers = tuple(marker for marker in ("910c", "ascend910_93", "910_93", "9382") if marker in evidence_text)
    warnings: list[str] = []
    errors: list[str] = []

    if target_fields["backend"] not in {"ascend", "npu"}:
        errors.append(
            "Triton target backend is " f"{target_fields['backend']!r}, expected 'ascend' or FlagTree's 'npu'."
        )
    if not cann_version.startswith("9.0"):
        message = f"Active CANN version is {cann_version!r} under {cann_root}; " "the formal target is CANN 9.0.x."
        (warnings if allow_non_cann9 else errors).append(message)
    if not evidence_markers:
        message = (
            "Could not prove the device is 910C from torch_npu/Triton/npu-smi. "
            "The generic 'Ascend910' display name alone is insufficient."
        )
        (warnings if allow_unverified_910c else errors).append(message)
    missing_symbols = [name for name, present in symbols.items() if not present]
    if missing_symbols:
        message = "Missing CANN ACLNN RNG symbols: " + ", ".join(missing_symbols)
        (errors if require_baseline else warnings).append(message)

    triton_ascend_version = _package_version("triton-ascend")
    torch_npu_version = _package_version("torch-npu") or _package_version("torch_npu")
    notes = [
        "The validated FlagTree image exposes Triton target backend 'npu'; "
        "package metadata may not use the triton-ascend distribution name."
    ]

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "device_index": device_index,
        "device_name": device_name,
        "device_properties": device_properties,
        "device_910c_evidence": list(evidence_markers),
        "triton_target": target_fields,
        "versions": {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_npu": torch_npu_version or "unknown",
            "triton": _package_version("triton") or "unknown",
            "triton_ascend": triton_ascend_version or "unknown",
            "cann": cann_version,
        },
        "cann_root": str(cann_root),
        "libopapi": str(opapi_path) if opapi_path else None,
        "aclnn_rng_symbols": symbols,
        "npu_smi": smi,
        "npu_smi_board": smi_board,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uname": list(platform.uname()),
        },
    }
    return result


def verify_source(project_root: Path) -> dict[str, Any]:
    paths = [
        path
        for base in (
            project_root / "src" / "flagrand",
            project_root / "benchmark" / "ascend_910c",
        )
        for path in base.rglob("*.py")
    ]
    paths.append(project_root / "benchmark" / "run_ascend_910c.py")
    paths = sorted(path for path in paths if path.is_file())
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "files": [path.relative_to(project_root).as_posix() for path in paths],
    }


def _resolve_cann_root() -> Path:
    candidates = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        (
            Path("/usr/local/Ascend/cann-9.0.0"),
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise RuntimeError("Cannot locate CANN. Source /usr/local/Ascend/cann-9.0.0/set_env.sh first.")


def _read_cann_version(cann_root: Path) -> str:
    candidates = (
        cann_root / "share/info/runtime/version.info",
        cann_root / "share/info/ascend_toolkit/version.info",
        cann_root / "version.info",
    )
    for path in candidates:
        if not path.is_file():
            continue
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        for key in ("Version", "version", "CANN_VERSION", "version_dir"):
            if values.get(key):
                return values[key]
        return path.read_text(encoding="utf-8", errors="replace").strip()[:200]
    name = cann_root.name.lower().removeprefix("cann-")
    return name if name and name != "latest" else "unknown"


def _find_libopapi(cann_root: Path) -> Path | None:
    candidates = (
        cann_root / "lib64/libopapi.so",
        cann_root / "lib64/stub/libopapi.so",
        cann_root / "aarch64-linux/lib64/libopapi.so",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    result = _run_optional(
        ("find", str(cann_root), "-name", "libopapi.so", "-type", "f", "-print", "-quit"),
        timeout=15,
    )
    first = result.get("stdout", "").strip().splitlines()
    return Path(first[0]).resolve() if first else None


def _probe_symbols(path: Path | None, names: tuple[str, ...]) -> dict[str, bool]:
    if path is None:
        return {name: False for name in names}
    try:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        return {name: hasattr(library, name) for name in names}
    except OSError:
        result = _run_optional(("nm", "-D", str(path)), timeout=20)
        text = result.get("stdout", "")
        return {name: name in text for name in names}


def _device_name(device_index: int) -> str:
    try:
        return str(torch.npu.get_device_name(device_index))
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def _device_properties(device_index: int) -> dict[str, Any]:
    try:
        props = torch.npu.get_device_properties(device_index)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    result = {"repr": repr(props)}
    for name in (
        "name",
        "total_memory",
        "multi_processor_count",
        "major",
        "minor",
    ):
        if hasattr(props, name):
            result[name] = _json_value(getattr(props, name))
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_optional(command: tuple[str, ...], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-4000:],
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
