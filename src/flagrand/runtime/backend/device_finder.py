from __future__ import annotations

import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version

import torch

from flagrand.runtime import error
from flagrand.runtime.common import _VENDOR_TORCH_ATTR, vendors

from . import get_vendor_info, get_vendor_infos


class DeviceDetector:
    """Process-wide accelerator detection following FlagGems' vendor flow."""

    _instance: "DeviceDetector | None" = None

    def __new__(cls, *args: object, **kwargs: object) -> "DeviceDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, vendor_name: str | None = None) -> None:
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        self.vendor_list = vendors.get_all_vendors().keys()
        self.info = self.get_vendor(vendor_name)
        self.vendor_name = self.info.vendor_name
        self.name = self.info.device_name
        self.vendor = vendors.get_all_vendors()[self.vendor_name]
        self.dispatch_key = self.name.upper() if self.info.dispatch_key is None else self.info.dispatch_key
        self.torch_device_fn = getattr(torch, self.name)
        self.device_count = self.torch_device_fn.device_count()
        self.support_fp64 = self.info.fp64_enabled
        self.support_bf16 = self.info.bf16_enabled
        self.support_int64 = self.info.int64_enabled

    def get_vendor(self, vendor_name: str | None = None):
        selected = self._get_vendor_from_env()
        if selected:
            return get_vendor_info(selected, query=True)
        if vendor_name:
            return get_vendor_info(vendor_name, query=True)
        selected = self._get_vendor_from_quick_cmd()
        if selected:
            return get_vendor_info(selected, query=True)
        selected = self._get_vendor_from_lib()
        if selected:
            return get_vendor_info(selected, query=True)
        return self._get_vendor_from_sys()

    def _get_vendor_from_env(self) -> str | None:
        if "PPU_SDK" in os.environ:
            return "thead"
        for key in ("RAND_VENDOR", "FLAGRAND_VENDOR", "RAND_BACKEND", "FLAGRAND_BACKEND"):
            if key in os.environ:
                return os.environ[key].lower()
        return None

    def _get_vendor_from_quick_cmd(self) -> str | None:
        try:
            import torch_npu

            torch_module = torch_npu
        except ImportError:
            torch_module = torch

        for vendor_name, attr in _VENDOR_TORCH_ATTR.items():
            if hasattr(torch_module, attr):
                return vendor_name

        if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "get_device_properties"):
            try:
                properties = torch_module.cuda.get_device_properties(0)
                if "NVIDIA" in properties.name.upper():
                    return "nvidia"
            except RuntimeError:
                pass
        return None

    def _get_vendor_from_lib(self) -> str | None:
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        backend_name = str(getattr(target, "backend", ""))
        target_vendors = {
            "ascend": "ascend",
            "npu": "ascend",
            "maca": "metax",
            "ppu": "thead",
            "hcu": "hygon",
            "musa": "mthreads",
        }
        if backend_name in target_vendors:
            return target_vendors[backend_name]
        if backend_name == "hip" and str(getattr(target, "arch", "")) == "gfx936":
            return "hygon"
        if backend_name == "cuda":
            try:
                if "corex" in version("triton").lower():
                    return "iluvatar"
            except PackageNotFoundError:
                pass
        return None

    def _get_vendor_from_sys(self):
        def check_vendor(info):
            if not info.device_query_cmd:
                return None
            try:
                result = subprocess.run(
                    shlex.split(info.device_query_cmd),
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                return None
            return info if result.returncode == 0 else None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_vendor, info) for info in get_vendor_infos()]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    return result
        error.device_not_found()

    def get_vendor_name(self) -> str:
        return self.vendor_name


__all__ = ["DeviceDetector"]
