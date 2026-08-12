from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

_COMPATIBILITY_INSTALLED = False


def install_ppu_compiler_compatibility() -> None:
    """Bridge FlagTree PPU 3.6 inline assembly to the installed PPU SDK."""
    global _COMPATIBILITY_INSTALLED

    if _COMPATIBILITY_INSTALLED:
        return

    from triton.backends.ppu import compiler as ppu_compiler

    if not isinstance(ppu_compiler.subprocess, _PpuCompilerSubprocessProxy):
        ppu_compiler.subprocess = _PpuCompilerSubprocessProxy(ppu_compiler.subprocess)

    # FlagTree PPU 3.6 references this name on compiler failure without
    # initializing it unless compile-log dumping is enabled.
    ppu_compiler.log_file = _compiler_log_path()
    _COMPATIBILITY_INSTALLED = True


class _PpuCompilerSubprocessProxy:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def run(self, command: Any, *args: Any, **kwargs: Any) -> Any:
        command_args = _command_args(command)
        if not command_args or Path(command_args[0]).name != "ppu-llc":
            return self._delegate.run(command, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="flagrand_ppu_llc_") as temporary_dir:
            rewritten_args, normalized_inputs = _normalize_compiler_inputs(
                command_args,
                Path(temporary_dir),
            )
            if not normalized_inputs:
                return self._delegate.run(command, *args, **kwargs)

            _record_normalization(normalized_inputs)
            rewritten_command: Any
            if isinstance(command, str):
                rewritten_command = shlex.join(rewritten_args)
            else:
                rewritten_command = rewritten_args
            return self._delegate.run(rewritten_command, *args, **kwargs)


def _command_args(command: Any) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, (list, tuple)):
        return [str(argument) for argument in command]
    return []


def _normalize_compiler_inputs(
    command_args: list[str],
    temporary_dir: Path,
) -> tuple[list[str], int]:
    rewritten_args = list(command_args)
    normalized_inputs = 0

    for index, argument in enumerate(command_args):
        source = Path(argument)
        if not argument.endswith(".tix.trans") or not source.is_file():
            continue

        original = source.read_text()
        normalized = _normalize_inline_assembly(original)
        if normalized == original:
            continue

        normalized_source = temporary_dir / source.name
        normalized_source.write_text(normalized)
        rewritten_args[index] = str(normalized_source)
        normalized_inputs += 1

    return rewritten_args, normalized_inputs


def _normalize_inline_assembly(source: str) -> str:
    return source.replace("ppu.", "")


def _record_normalization(normalized_inputs: int) -> None:
    path = os.environ.get("FLAGRAND_PPU_LLC_COMPAT_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as output:
        output.write(f"normalized_inputs={normalized_inputs}\n")


def _compiler_log_path() -> str:
    configured = os.environ.get("FLAGRAND_PPU_COMPILE_LOG")
    path = (
        Path(configured)
        if configured
        else Path(os.environ.get("TRITON_CACHE_DIR", "/tmp")) / ("flagrand_ppu_compile.log")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
