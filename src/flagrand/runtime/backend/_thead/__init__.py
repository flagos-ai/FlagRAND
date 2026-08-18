from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="thead",
    device_name="cuda",
    device_query_cmd="ppu-smi",
    trademark="T-Head Zhenwu PPU",
    fp64_enabled=False,
    backend_name="ppu",
    cached_binary_launch=True,
    portable_f32_math=True,
)

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()


def activate() -> None:
    from flagrand.runtime.ppu_compiler import install_ppu_compiler_compatibility

    install_ppu_compiler_compatibility()


__all__ = ["CUSTOMIZED_UNUSED_OPS", "activate", "vendor_info"]
