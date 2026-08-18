from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor


vendor_info = VendorDescriptor(
    vendor_name="ascend",
    device_name="npu",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
    triton_extra_name="ascend",
    trademark="Huawei Ascend",
    backend_name="npu",
    cached_binary_launch=True,
    portable_f32_math=True,
)

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()

__all__ = ["CUSTOMIZED_UNUSED_OPS", "vendor_info"]
