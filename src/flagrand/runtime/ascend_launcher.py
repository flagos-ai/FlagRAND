"""Prepared launches for the Ascend Triton runtime.

The initialized NPULauncher object is retained, not replaced by its C entry.
Its compile-only, tensor-registration and profiler bookkeeping remain active.
The public JIT still handles cold specializations and dynamic pre-run modes.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass


class _EnvironmentModes:
    """Observe both environment values on every call, parsing only changes.

    The changed-value mode targets the installed Linux CPython os._Environ:
    its byte dictionary is the backing store read by os.environ.get. A changed
    value is read through that public method, retaining torch_npu's log-once
    wrapper and the original string normalization (including Unicode spaces).
    """

    def __init__(self, changed_values=True):
        self.changed_values = changed_values
        self._data = os.environ._data
        self._portable_key = os.fsencode("FLAGRAND_FORCE_PORTABLE")
        self._debug_key = os.fsencode("TRITON_DEBUG")
        self._last = object()
        self._modes = (False, False)

    def __call__(self):
        if self.changed_values:
            observed = (
                self._data.get(self._portable_key),
                self._data.get(self._debug_key),
            )
            if observed == self._last:
                return self._modes
            self._last = observed
        portable = os.environ.get("FLAGRAND_FORCE_PORTABLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        debug = os.environ.get("TRITON_DEBUG", "0") == "1"
        self._modes = portable, debug
        return self._modes


@dataclass
class _Entry:
    compiled: object
    device: int
    run: object
    function: object
    packed_metadata: object
    launch_metadata: object
    hook_class: object


class PreparedAscendLauncher:
    """Retain initialized launch handles and resolve each call's stream."""

    def __init__(
        self,
        kernel,
        *,
        constexpr_names,
        option_names=("num_warps",),
        max_entries=32,
        environment_modes=None,
        entry_factory=None,
    ):
        from triton.runtime.driver import driver

        self.kernel = kernel
        self.entry_factory = entry_factory
        self.constexpr_names = tuple(constexpr_names)
        self.option_names = tuple(option_names)
        self.max_entries = max_entries
        self.environment_modes = environment_modes or _EnvironmentModes()
        self.driver = driver
        self.entries = OrderedDict()
        self.last_key = object()
        self.last_entry = None

    def _regular(self, grid, args, constants, options):
        kwargs = dict(zip(self.constexpr_names, constants))
        kwargs.update(zip(self.option_names, options))
        return self.kernel[grid](*args, **kwargs)

    def launch(self, grid, args, constexpr_values, option_values, *, specialization_key):
        portable, debug = self.environment_modes()
        grid3 = grid + (1,) * (3 - len(grid))
        if portable or debug or self.kernel.pre_run_hooks:
            return self._regular(grid3, args, constexpr_values, option_values)
        # Callers include dtype, device, pointer alignment, scalar specialization
        # and constexpr values in the key; launch options remain distinct here.
        key = (specialization_key, option_values)
        if key == self.last_key:
            entry = self.last_entry
        else:
            entry = self.entries.get(key)
            if entry is None:
                compiled = self._regular(grid3, args, constexpr_values, option_values)
                if compiled is None:
                    # An installed JIT compilation hook may intentionally return
                    # no kernel. The regular launch has already honored it.
                    return compiled
                # The regular run initialized handles. Reading run here uses
                # its actual lazy accessor once; hot calls retain that object.
                device = self.driver.active.get_current_device()
                run = compiled.run
                if self.entry_factory is not None:
                    run = self.entry_factory(compiled, run)
                entry = _Entry(
                    compiled,
                    device,
                    run,
                    compiled.function,
                    compiled.packed_metadata,
                    compiled.launch_metadata,
                    type(compiled),
                )
                self.entries[key] = entry
                self.entries.move_to_end(key)
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
                self.last_key, self.last_entry = key, entry
                return compiled
            self.entries.move_to_end(key)
            self.last_key, self.last_entry = key, entry
        stream = self.driver.active.get_current_stream(entry.device)
        # The installed JIT reads these class hooks each invocation. Do not
        # cache the current hook values or the runtime launch metadata.
        enter = entry.hook_class.launch_enter_hook
        leave = entry.hook_class.launch_exit_hook
        metadata = entry.launch_metadata(grid3, stream, *args) if enter is not None else None
        entry.run(
            grid3[0],
            grid3[1],
            grid3[2],
            stream,
            entry.function,
            entry.packed_metadata,
            metadata,
            enter,
            leave,
            *args,
        )
        return entry.compiled
