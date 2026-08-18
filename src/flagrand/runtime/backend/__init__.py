from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from types import ModuleType

from flagrand.runtime.common import vendors

from .backend_utils import BackendEventBase, VendorDescriptor


def _module_functions(module: ModuleType | None) -> list[tuple[str, object]]:
    if module is None:
        return []
    return [
        (name, fn) for name, fn in inspect.getmembers(module, inspect.isfunction) if fn.__module__ == module.__name__
    ]


class BackendState(BackendEventBase):
    _instance: "BackendState | None" = None

    def __new__(cls) -> "BackendState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        self.vendor_module: ModuleType | None = None
        self.vendor_info: VendorDescriptor | None = None
        self.device_name: str | None = None
        self.torch_device_object: object | None = None
        self.ops_module: ModuleType | None = None
        self.fused_module: ModuleType | None = None
        self.customized_ops: list[tuple[str, object]] | None = None

    def is_available(self) -> bool:
        return self.vendor_module is not None

    def get_ops(self, vendor: str | None = None) -> list[tuple[str, object]]:
        return get_customized_ops(vendor)


_state = BackendState()


def get_vendor_module(vendor_name: str, *, query: bool = False) -> ModuleType:
    module = importlib.import_module(f"{__package__}._{vendor_name}")
    if not query:
        _state.vendor_module = module
    return module


def get_vendor_info(vendor_name: str | None = None, *, query: bool = False) -> VendorDescriptor:
    if vendor_name is None:
        if _state.vendor_info is None:
            raise RuntimeError("FlagRAND backend has not been activated.")
        return _state.vendor_info
    return get_vendor_module(vendor_name, query=query).vendor_info


def get_vendor_infos() -> list[VendorDescriptor]:
    return [get_vendor_info(vendor_name, query=True) for vendor_name in vendors.get_all_vendors()]


def activate_backend(vendor_name: str) -> None:
    module = get_vendor_module(vendor_name)
    info = module.vendor_info
    _state.vendor_info = info
    _state.device_name = info.device_name
    _state.torch_device_object = None
    _state.ops_module = None
    _state.fused_module = None
    _state.customized_ops = None
    activate = getattr(module, "activate", None)
    if activate is not None:
        activate()


def gen_torch_device_object(vendor_name: str | None = None):
    if _state.torch_device_object is None:
        info = get_vendor_info(vendor_name, query=vendor_name is not None)
        import torch

        _state.torch_device_object = getattr(torch, info.device_name)
    return _state.torch_device_object


def import_vendor_extra_lib(vendor_name: str | None = None) -> None:
    selected = vendor_name or get_vendor_info().vendor_name
    package = f"{__package__}._{selected}"
    try:
        _state.ops_module = importlib.import_module(f"{package}.ops")
    except ModuleNotFoundError as error:
        if error.name != f"{package}.ops":
            raise
    try:
        _state.fused_module = importlib.import_module(f"{package}.fused")
    except ModuleNotFoundError as error:
        if error.name != f"{package}.fused":
            raise


def get_customized_ops(vendor_name: str | None = None) -> list[tuple[str, object]]:
    if _state.customized_ops is None:
        import_vendor_extra_lib(vendor_name)
        _state.customized_ops = _module_functions(_state.ops_module) + _module_functions(_state.fused_module)
    return _state.customized_ops


class BackendArchEvent(BackendEventBase):
    def __init__(self) -> None:
        self.module: ModuleType | None = None
        vendor_module = _state.vendor_module
        if vendor_module is None or not hasattr(vendor_module, "ARCH_MAP"):
            return
        properties = gen_torch_device_object().get_device_properties(0)
        keys = [f"{properties.major}.{properties.minor}", str(properties.major)]
        arch = next((vendor_module.ARCH_MAP[key] for key in keys if key in vendor_module.ARCH_MAP), None)
        if arch is None:
            return
        module_name = f"{vendor_module.__name__}.{arch}.ops"
        module_path = Path(vendor_module.__path__[0]) / arch / "ops.py"
        if module_path.exists():
            self.module = importlib.import_module(module_name)

    def is_available(self) -> bool:
        return self.module is not None

    def get_ops(self, vendor: str | None = None) -> list[tuple[str, object]]:
        return _module_functions(self.module)


class TritonVersionEvent(BackendEventBase):
    def __init__(self) -> None:
        self.module: ModuleType | None = None
        if _state.vendor_module is None:
            return
        import triton

        version_dir = "triton_" + triton.__version__.replace(".", "_").replace("+", "_")
        module_path = Path(_state.vendor_module.__path__[0]) / version_dir / "ops.py"
        if module_path.exists():
            self.module = importlib.import_module(f"{_state.vendor_module.__name__}.{version_dir}.ops")

    def is_available(self) -> bool:
        return self.module is not None

    def get_ops(self, vendor: str | None = None) -> list[tuple[str, object]]:
        return _module_functions(self.module)


class SpecOpRegistrar:
    def __init__(self, registry: dict[str, object], vendor: str | None = None) -> None:
        self._globals = registry
        self.vendor = vendor

    def apply(self, vendor: str | None = None) -> None:
        selected = vendor or self.vendor
        for event in (_state, BackendArchEvent(), TritonVersionEvent()):
            if event.is_available():
                for name, function in event.get_ops(selected):
                    self._globals[name] = function


def current_backend() -> str:
    return get_vendor_info().backend_name


def current_vendor() -> str:
    return get_vendor_info().vendor_name


def force_portable_backend() -> bool:
    return os.environ.get("FLAGRAND_FORCE_PORTABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def supports_cached_binary_launch(backend: str | None = None) -> bool:
    if force_portable_backend():
        return False
    if backend is None:
        return get_vendor_info().cached_binary_launch
    return backend in {"cuda", "ppu", "npu"}


def uses_portable_f32_math(backend: str | None = None) -> bool:
    if force_portable_backend():
        return True
    if backend is None:
        return get_vendor_info().portable_f32_math
    return backend != "cuda"


def get_unused_ops(vendor_name: str | None = None) -> list[str]:
    module = _state.vendor_module if vendor_name is None else get_vendor_module(vendor_name, query=True)
    return list(module.CUSTOMIZED_UNUSED_OPS)


def get_backend_state() -> BackendState:
    return _state


__all__ = [
    "BackendArchEvent",
    "BackendState",
    "SpecOpRegistrar",
    "TritonVersionEvent",
    "activate_backend",
    "current_backend",
    "current_vendor",
    "force_portable_backend",
    "gen_torch_device_object",
    "get_backend_state",
    "get_customized_ops",
    "get_unused_ops",
    "get_vendor_info",
    "get_vendor_infos",
    "get_vendor_module",
    "supports_cached_binary_launch",
    "uses_portable_f32_math",
]
