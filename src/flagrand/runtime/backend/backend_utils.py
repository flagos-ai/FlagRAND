from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VendorDescriptor:
    """Metadata and runtime policy supplied by one accelerator backend."""

    vendor_name: str
    device_name: str
    device_query_cmd: str
    dispatch_key: str | None = None
    triton_extra_name: str | None = None
    trademark: str | None = None
    fp64_enabled: bool = True
    bf16_enabled: bool = True
    int64_enabled: bool = True
    backend_name: str = "cuda"
    cached_binary_launch: bool = True
    portable_f32_math: bool = False


class BackendEventBase:
    def get_ops(self, vendor: str | None = None) -> list[tuple[str, object]]:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


__all__ = ["BackendEventBase", "VendorDescriptor"]
