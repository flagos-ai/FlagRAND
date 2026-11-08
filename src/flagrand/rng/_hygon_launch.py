"""Prepared HCU launches using its existing zero-scratch C entry point."""

import os
from dataclasses import dataclass
from types import BuiltinFunctionType

from triton import knobs
from triton.runtime.driver import driver


@dataclass
class Call:
    compiled: object
    run: object
    cooperative: bool
    function: object
    metadata: object
    launch_metadata: object
    device: int
    stream_reader: object
    portable: object
    debug: object


class HygonPreparedLaunch:
    def __init__(self, launcher, grid):
        self.prepared = launcher.prepare(grid)
        self.kernel = launcher._kernel
        self.grid = self.prepared._grid
        self.call = None
        self.key = None

    def launch(self, args, constants, options, *, specialization_key):
        call = self.call
        runtime = knobs.runtime
        kernel = self.kernel
        if (
            call is not None
            and self.key is specialization_key
            and os.environ.get("FLAGRAND_FORCE_PORTABLE") == call.portable
            and kernel.debug == call.debug
            and not kernel.pre_run_hooks
            and not runtime.launch_enter_hook.calls
            and not runtime.launch_exit_hook.calls
            and not knobs.compilation.instrumentation_mode
            and runtime.add_stages_inspection_hook is None
            and kernel.launch_metadata is None
        ):
            # HCU's HIPLauncher prepends cooperative-grid and profile-scratch
            # arguments. Tensor conversion remains in the original C launcher.
            stream = call.stream_reader(call.device)
            call.run(
                call.cooperative,
                *self.grid,
                stream,
                call.function,
                None,
                call.metadata,
                call.launch_metadata,
                None,
                None,
                *(args + constants),
            )
            return call.compiled
        self.call = None
        compiled = self.prepared.launch(args, constants, options, specialization_key=specialization_key)
        entry = self.prepared._entry
        if entry is not None:
            launch = compiled.run
            if launch.profile_scratch_size == 0 and isinstance(launch.launch, BuiltinFunctionType):
                self.call = Call(
                    compiled,
                    launch.launch,
                    launch.launch_cooperative_grid,
                    compiled.function,
                    compiled.packed_metadata,
                    entry.launch_metadata,
                    entry.device,
                    driver.active.get_current_stream,
                    os.environ.get("FLAGRAND_FORCE_PORTABLE"),
                    kernel.debug,
                )
                self.key = specialization_key
        return compiled


def prepare_hygon(launcher, grid):
    return HygonPreparedLaunch(launcher, grid)
