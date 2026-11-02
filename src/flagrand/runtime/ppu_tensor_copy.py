"""Load the PPU Tensor/Stream copy binding on its first buffered use."""

from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def load_ppu_tensor_copy():
    import torch
    from torch.utils.cpp_extension import load

    sdk = Path(os.environ.get("FLAGRAND_PPU_SDK_ROOT", "/usr/local/PPU_SDK")).resolve()
    runtime = sdk / "targets/x86_64-linux/lib/libhggcrt.12.0.so"
    torch_library = Path(torch.__file__).resolve().parent / "lib"
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    return load(
        name="_flagrand_ppu_tensor_copy",
        sources=[str(Path(__file__).with_name("csrc") / "ppu_tensor_copy.cpp")],
        extra_include_paths=[str(sdk / "CUDA_SDK/include"), str(sdk / "targets/x86_64-linux/include")],
        extra_cflags=["-O3", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"],
        extra_ldflags=[
            "-lc10_cuda",
            "-ltorch_cuda",
            str(runtime),
            f"-Wl,-rpath,{runtime.parent}",
            f"-Wl,-rpath,{torch_library}",
        ],
        with_cuda=False,
        verbose=False,
    )


if __name__ == "__main__":
    print(load_ppu_tensor_copy().__file__)
