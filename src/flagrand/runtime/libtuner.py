from __future__ import annotations

import os
from typing import Any

import triton


def use_libtuner() -> bool:
    value = os.environ.get("FLAGRAND_USE_LIBTUNER", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


try:
    from flag_gems.utils.libentry import libentry, libtuner

    HAS_LIBTUNER = True
    LIBTUNER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional FlagGems install.
    libentry = None
    libtuner = None
    HAS_LIBTUNER = False
    LIBTUNER_IMPORT_ERROR = exc


PHILOX_LIBTUNER_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps, num_stages=1)
    for block in (128, 256, 512, 1024)
    for warps in (2, 4, 8)
]


def require_libtuner() -> None:
    if HAS_LIBTUNER:
        return
    detail = (
        f" Import failed with {type(LIBTUNER_IMPORT_ERROR).__name__}: "
        f"{LIBTUNER_IMPORT_ERROR}"
        if LIBTUNER_IMPORT_ERROR is not None
        else ""
    )
    raise RuntimeError(
        "FLAGRAND_USE_LIBTUNER=1 requires FlagGems LibTuner "
        "(flag_gems.utils.libentry)."
        + detail
    )


def libtuner_status() -> dict[str, Any]:
    return {
        "requested": use_libtuner(),
        "available": HAS_LIBTUNER,
        "import_error": (
            None
            if LIBTUNER_IMPORT_ERROR is None
            else f"{type(LIBTUNER_IMPORT_ERROR).__name__}: {LIBTUNER_IMPORT_ERROR}"
        ),
        "configs": [
            {
                "BLOCK": config.kwargs["BLOCK"],
                "num_warps": getattr(config, "num_warps", None),
                "num_stages": getattr(config, "num_stages", None),
            }
            for config in PHILOX_LIBTUNER_CONFIGS
        ],
    }
