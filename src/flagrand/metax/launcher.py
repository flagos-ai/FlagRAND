from __future__ import annotations

from collections import OrderedDict
from typing import Any

from triton.runtime.driver import driver


class MetaXKernelLauncher:
    """Cache MACA Triton kernels and launch their compiled binary directly."""

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
        self._entries: OrderedDict[object, Any] = OrderedDict()

    def launch(
        self,
        grid: int | tuple[int, ...],
        args: tuple[object, ...],
        constexpr_values: tuple[object, ...],
        option_values: tuple[object, ...],
        *,
        specialization_key: object | None = None,
    ) -> Any:
        del specialization_key
        grid3 = _normalize_grid(grid)
        kwargs = dict(zip(self._constexpr_names, constexpr_values))
        kwargs.update(zip(self._option_names, option_values))

        if not hasattr(self._kernel, "binder"):
            return self._kernel[grid3](*args, **kwargs)

        if self._kernel.binder is None:
            self._kernel.create_binder()

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
        compiled = self._entries.get(key)
        if compiled is None:
            compiled = self._kernel[grid3](*args, **kwargs)
            compiled._init_handles()
            self._entries[key] = compiled
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return compiled

        self._entries.move_to_end(key)
        stream = driver.active.get_current_stream(device)
        launch_metadata = compiled.launch_metadata(grid3, stream, *non_constexpr_values)
        compiled.run(
            grid3[0],
            grid3[1],
            grid3[2],
            stream,
            compiled.function,
            compiled.packed_metadata,
            launch_metadata,
            compiled.launch_enter_hook,
            compiled.launch_exit_hook,
            *non_constexpr_values,
        )
        return compiled


def _normalize_grid(grid: int | tuple[int, ...]) -> tuple[int, int, int]:
    if isinstance(grid, int):
        return (grid, 1, 1)
    if not 1 <= len(grid) <= 3:
        raise ValueError(f"Triton grid must have one to three dimensions, got {grid!r}")
    return (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)


__all__ = ["MetaXKernelLauncher"]
