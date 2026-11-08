"""Build the CoreX Tensor copy/cache binding from this repository's source."""

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_iluvatar_tensor_copy():
    import torch
    from torch.utils.cpp_extension import load

    sdk = Path(os.environ.get("FLAGRAND_COREX_SDK_ROOT", "/usr/local/corex")).resolve()
    torch_library = Path(torch.__file__).resolve().parent / "lib"
    return load(
        name="_flagrand_iluvatar_tensor_copy",
        sources=[str(Path(__file__).with_name("csrc") / "iluvatar_tensor_copy.cpp")],
        extra_include_paths=[str(sdk / "include")],
        extra_cflags=["-O3"],
        extra_ldflags=[
            "-lc10_cuda",
            "-ltorch_cuda",
            f"-L{sdk / 'lib64'}",
            "-lcudart",
            f"-Wl,-rpath,{sdk / 'lib64'}",
            f"-Wl,-rpath,{torch_library}",
        ],
        with_cuda=False,
        verbose=False,
    )


if __name__ == "__main__":
    print(load_iluvatar_tensor_copy().__file__)
