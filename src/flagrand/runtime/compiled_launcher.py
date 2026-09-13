from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

try:
    from triton import knobs
except ImportError:
    knobs = None
from triton.runtime.driver import driver

from flagrand.runtime.backend import supports_cached_binary_launch
from flagrand.runtime.specialization import bind_specialization, find_matching_entry, value_tokens


@dataclass
class _LaunchEntry:
    compiled: Any
    device: int
    binder: Any
    specialization: tuple[object, ...]
    parsed_options: object
    value_tokens: tuple[tuple[object, ...], ...]
    launch_metadata: object


@dataclass
class _LegacyLaunchEntry:
    compiled: Any
    device: int


class CachedKernelLauncher:
    """Reuse a compiled Triton specialization while resolving each call's stream."""

    def __init__(
        self,
        kernel: Any,
        *,
        constexpr_names: tuple[str, ...],
        option_names: tuple[str, ...] = ("num_warps",),
        max_entries: int = 32,
    ) -> None:
        self._kernel = kernel
        self._constexpr_names = constexpr_names
        self._option_names = option_names
        self._max_entries = max_entries
        self._entries: list[_LaunchEntry] = []
        self._keyed_entries: OrderedDict[object, _LaunchEntry] = OrderedDict()
        self._legacy_entries: OrderedDict[object, _LegacyLaunchEntry] = OrderedDict()
        self._legacy_keyed_entries: OrderedDict[object, _LegacyLaunchEntry] = OrderedDict()

    def launch(
        self,
        grid: int | tuple[int, ...],
        args: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
        *,
        specialization_key: object | None = None,
    ) -> Any:
        grid3 = _normalize_grid(grid)
        if supports_cached_binary_launch() and knobs is None and getattr(self._kernel, "binder", None) is not None:
            return self._launch_legacy(
                grid3,
                args,
                constexpr_values,
                option_values,
                specialization_key,
            )
        values = args + constexpr_values
        if not _requires_regular_jit(self._kernel):
            if specialization_key is not None:
                entry = self._keyed_entries.get(specialization_key)
                if entry is not None:
                    self._keyed_entries.move_to_end(specialization_key)
                    return _launch_entry(entry, grid3, values)
            entry = find_matching_entry(
                self._entries,
                values,
                self._option_names,
                option_values,
            )
            if entry is not None:
                if self._entries[0] is not entry:
                    for index, candidate in enumerate(self._entries[1:], start=1):
                        if candidate is entry:
                            self._entries.insert(0, self._entries.pop(index))
                            break
                self._remember_key(specialization_key, entry)
                return _launch_entry(entry, grid3, values)
        return self._compile_and_launch(
            grid3,
            args,
            values,
            constexpr_values,
            option_values,
            specialization_key,
        )

    def _launch_legacy(
        self,
        grid: tuple[int, int, int],
        args: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
        specialization_key: object | None,
    ) -> Any:
        if specialization_key is not None:
            entry = self._legacy_keyed_entries.get(specialization_key)
            if entry is not None:
                self._legacy_keyed_entries.move_to_end(specialization_key)
                return _launch_legacy_entry(entry, grid, args)

        kwargs = dict(zip(self._constexpr_names, constexpr_values))
        kwargs.update(zip(self._option_names, option_values))
        kwargs["debug"] = getattr(self._kernel, "debug", False)
        _, sig_and_spec, bound_constexprs, non_constexpr_values, excess_kwargs = self._kernel.binder(
            *args,
            **kwargs,
        )
        device = driver.active.get_current_device()
        key = (
            device,
            tuple(sig_and_spec),
            tuple(bound_constexprs),
            tuple(sorted(excess_kwargs.items())),
        )
        entry = self._legacy_entries.get(key)
        if entry is None:
            compiled = self._kernel[grid](*args, **kwargs)
            compiled._init_handles()
            entry = _LegacyLaunchEntry(compiled=compiled, device=device)
            self._legacy_entries[key] = entry
            self._legacy_entries.move_to_end(key)
            while len(self._legacy_entries) > self._max_entries:
                self._legacy_entries.popitem(last=False)
            self._remember_legacy_key(specialization_key, entry)
            return compiled

        self._legacy_entries.move_to_end(key)
        self._remember_legacy_key(specialization_key, entry)
        return _launch_legacy_entry(entry, grid, non_constexpr_values)

    def _remember_legacy_key(self, key: object | None, entry: _LegacyLaunchEntry) -> None:
        if key is None:
            return
        self._legacy_keyed_entries[key] = entry
        self._legacy_keyed_entries.move_to_end(key)
        while len(self._legacy_keyed_entries) > self._max_entries:
            self._legacy_keyed_entries.popitem(last=False)

    def _compile_and_launch(
        self,
        grid3: tuple[int, int, int],
        args: tuple[object, ...],
        values: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
        specialization_key: object | None,
    ) -> Any:
        if len(constexpr_values) != len(self._constexpr_names):
            raise ValueError("constexpr value count does not match launcher configuration")
        if len(option_values) != len(self._option_names):
            raise ValueError("option value count does not match launcher configuration")

        kwargs = dict(zip(self._constexpr_names, constexpr_values))
        kwargs.update(zip(self._option_names, option_values))
        compiled = self._kernel[grid3](*args, **kwargs)
        if compiled is None or _requires_regular_jit(self._kernel):
            return compiled

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        compiled._init_handles()
        binder = self._kernel.device_caches[device][4]
        specialization, parsed_options = bind_specialization(
            binder,
            values,
            self._option_names,
            option_values,
        )
        entry = _LaunchEntry(
            compiled=compiled,
            device=device,
            binder=binder,
            specialization=specialization,
            parsed_options=parsed_options,
            value_tokens=value_tokens(values),
            launch_metadata=compiled.launch_metadata(grid3, stream, *values),
        )
        self._entries.insert(0, entry)
        del self._entries[self._max_entries :]
        self._remember_key(specialization_key, entry)
        return compiled

    def _remember_key(self, key: object | None, entry: _LaunchEntry) -> None:
        if key is None:
            return
        self._keyed_entries[key] = entry
        self._keyed_entries.move_to_end(key)
        while len(self._keyed_entries) > self._max_entries:
            self._keyed_entries.popitem(last=False)


def _launch_entry(
    entry: _LaunchEntry,
    grid: tuple[int, int, int],
    values: tuple[object, ...],
) -> Any:
    compiled = entry.compiled
    stream = driver.active.get_current_stream(entry.device)
    compiled.run(
        grid[0],
        grid[1],
        grid[2],
        stream,
        compiled.function,
        compiled.packed_metadata,
        entry.launch_metadata,
        None,
        None,
        *values,
    )
    return compiled


def _launch_legacy_entry(
    entry: _LegacyLaunchEntry,
    grid: tuple[int, int, int],
    values: tuple[object, ...],
) -> Any:
    compiled = entry.compiled
    stream = driver.active.get_current_stream(entry.device)
    launch_enter_hook = getattr(compiled, "launch_enter_hook", None)
    launch_exit_hook = getattr(compiled, "launch_exit_hook", None)
    launch_metadata = compiled.launch_metadata(grid, stream, *values) if launch_enter_hook is not None else None
    compiled.run(
        grid[0],
        grid[1],
        grid[2],
        stream,
        compiled.function,
        compiled.packed_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *values,
    )
    return compiled


def _requires_regular_jit(kernel: Any) -> bool:
    if knobs is None:
        return True
    runtime_knobs = getattr(knobs, "runtime", None)
    compilation_knobs = getattr(knobs, "compilation", None)
    launch_enter_hook = getattr(runtime_knobs, "launch_enter_hook", None)
    launch_exit_hook = getattr(runtime_knobs, "launch_exit_hook", None)
    return bool(
        not supports_cached_binary_launch()
        or kernel.pre_run_hooks
        or getattr(launch_enter_hook, "calls", ())
        or getattr(launch_exit_hook, "calls", ())
        or getattr(compilation_knobs, "instrumentation_mode", False)
        or getattr(runtime_knobs, "add_stages_inspection_hook", None) is not None
        or kernel.launch_metadata is not None
    )


def _normalize_grid(grid: int | tuple[int, ...]) -> tuple[int, int, int]:
    if isinstance(grid, int):
        return (grid, 1, 1)
    if not 1 <= len(grid) <= 3:
        raise ValueError(f"Triton grid must have one to three dimensions, got {grid!r}")
    return (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
