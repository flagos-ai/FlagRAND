from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import triton


def verify_preview_source(project_root: Path) -> dict[str, Any]:
    import flagrand

    source_root = (project_root / "src").resolve()
    imported_file = Path(flagrand.__file__).resolve()
    try:
        imported_file.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported flagrand from {imported_file}, expected a file under {source_root}."
        ) from exc

    source_files = sorted((source_root / "flagrand").rglob("*.py"))
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "imported_file": str(imported_file),
        "source_root": str(source_root),
        "python_file_count": len(source_files),
        "source_sha256": digest.hexdigest(),
    }


def apply_maca_compatibility() -> dict[str, Any]:
    backend, target = _current_target()
    applied: list[str] = []
    if backend == "maca":
        from flagrand.runtime import (
            supports_cached_binary_launch,
            uses_portable_f32_math,
        )

        if supports_cached_binary_launch() or not uses_portable_f32_math():
            raise RuntimeError("FlagRand did not select its built-in MACA fallbacks.")
        applied.extend(
            (
                "core_standard_triton_jit_launcher",
                "core_portable_tl_sin_cos_box_muller_f32",
            )
        )
    return {
        "backend": backend,
        "target": target,
        "applied": applied,
    }


def collect_environment() -> dict[str, Any]:
    backend, target = _current_target()
    device: dict[str, Any] = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        device = {
            "name": getattr(props, "name", "unknown"),
            "total_memory": getattr(props, "total_memory", None),
            "major": getattr(props, "major", None),
            "minor": getattr(props, "minor", None),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "triton_backend": backend,
        "triton_target": target,
        "device": device,
    }


def _current_target() -> tuple[str, str]:
    try:
        from triton.runtime import driver

        target = driver.active.get_current_target()
        return str(getattr(target, "backend", "unknown")), str(target)
    except Exception as exc:
        return "unavailable", f"{type(exc).__name__}: {exc}"
