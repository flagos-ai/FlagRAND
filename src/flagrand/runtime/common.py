from __future__ import annotations

from enum import Enum


class vendors(Enum):
    NVIDIA = 0
    METAX = 1
    ILUVATAR = 2
    ASCEND = 3
    THEAD = 4
    HYGON = 5
    MTHREADS = 6

    @classmethod
    def get_all_vendors(cls) -> dict[str, "vendors"]:
        return {member.name.lower(): member for member in cls}


_VENDOR_TORCH_ATTR = {
    "ascend": "npu",
    "iluvatar": "corex",
    "mthreads": "musa",
}


__all__ = ["_VENDOR_TORCH_ATTR", "vendors"]
