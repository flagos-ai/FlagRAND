"""Select the RNG reference shipped by the active CoreX toolkit."""

import os
from functools import lru_cache
from pathlib import Path


def stock_library_override() -> Path | None:
    for variable in ("FLAGRAND_COREX_CURAND_PATH", "CURAND_LIBRARY_PATH"):
        if value := os.environ.get(variable):
            return Path(value)
    return None


def sdk_root() -> Path:
    # A job may keep COREX_HOME from its image and select a different toolkit's
    # stock library. Use the headers paired with the library actually loaded.
    override = stock_library_override()
    if override is not None:
        root = override.resolve().parent.parent
        if (root / "include/curand_kernel.h").is_file():
            return root
    for variable in ("FLAGRAND_COREX_SDK_ROOT", "COREX_HOME", "COREX_ROOT"):
        if value := os.environ.get(variable):
            return Path(value).resolve()
    root = Path("/usr/local/corex")
    legacy = Path("/usr/local/corex-4.4.0")
    return (legacy if not root.exists() and legacy.exists() else root).resolve()


@lru_cache(maxsize=None)
def has_ixrand_headers(root: Path) -> bool:
    # CoreX 4.5 forwards the public device API to IX; 4.4 defines it inline.
    header = root / "include/curand_kernel.h"
    if not header.is_file():
        return False
    with header.open("rb") as stream:
        return b"IX/curand_kernel.h" in stream.read(4096)


def needs_corrected_reference(engine: str, api: str) -> bool:
    return (engine == "mrg32k3a" or api == "poisson") and not has_ixrand_headers(sdk_root())
