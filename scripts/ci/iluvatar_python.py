"""Reuse the CoreX image's Python stack or select a published SDK wheel pair."""

import os
import re
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path


def toolkit_version(root):
    compiler = root / "bin/clang++"
    if compiler.is_file():
        version = subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0]
        match = re.search(r"\((\d+\.\d+\.\d+(?:\.\d+)?)(?=[ )])", version)
        if match:
            return match.group(1)
    return None


def image_paths(sdk_version):
    try:
        torch_dist = metadata.distribution("torch")
    except metadata.PackageNotFoundError:
        return None
    match = re.search(r"\+corex\.(\d+\.\d+\.\d+(?:\.\d+)?)", torch_dist.version)
    if not match:
        return None
    wheel_sdk = match.group(1)
    if sdk_version and wheel_sdk != sdk_version and not wheel_sdk.startswith(f"{sdk_version}."):
        print(f"Image Torch uses CoreX {wheel_sdk}; active toolkit is {sdk_version}", file=sys.stderr)
        return None
    try:
        import torch
        import triton
    except ImportError:
        return None
    if Path(torch_dist.locate_file("torch/__init__.py")).resolve() != Path(torch.__file__).resolve():
        return None
    triton_file = Path(triton.__file__).resolve()
    for name in ("triton", "flagtree", "flagtree-iluvatar"):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        if name == "triton" and dist.version.partition("+corex.")[2] != wheel_sdk:
            continue
        if name == "flagtree" and "iluvatar" not in dist.version.lower():
            continue
        if Path(dist.locate_file("triton/__init__.py")).resolve() == triton_file:
            print(f"Reusing {sys.executable}: torch {torch_dist.version}, {name} {dist.version}", file=sys.stderr)
            return list(
                dict.fromkeys(
                    [
                        str(Path(torch.__file__).resolve().parent.parent),
                        str(triton_file.parent.parent),
                        *(p for p in sys.path if p.endswith(("/site-packages", "/dist-packages"))),
                    ]
                )
            )
    return None


def image_venv(target, paths):
    if target.resolve() == Path(sys.prefix).resolve() and not target.is_symlink():
        return
    # uv --clear follows directory symlinks; leave an external image environment intact.
    source = Path(sys.prefix).resolve() / "bin/python" if target.is_symlink() else Path(sys.executable)
    if target.is_symlink():
        target.unlink()
    subprocess.run(
        ["uv", "venv", "--clear", "--system-site-packages", "--python", str(source), str(target)], check=True
    )
    # A child venv inherits system packages but not packages in its parent venv.
    purelib = sysconfig.get_path("purelib", vars={"base": str(target), "platbase": str(target)})
    (Path(purelib) / "flagrand-corex-image.pth").write_text("\n".join(paths) + "\n")


if __name__ == "__main__":
    target = Path(sys.argv[1])
    root = Path(os.environ["COREX_HOME"])
    version = toolkit_version(root)
    paths = image_paths(version)
    if paths:
        image_venv(target, paths)
        print("image")
    else:
        if version is None and not root.exists():
            version = "4.4.0"
        if version not in ("4.4.0", "4.5.0.20260804"):
            raise SystemExit(
                f"No preset wheels for CoreX {version or root}. Set FLAGRAND_SYSTEM_PYTHON to its image Python."
            )
        if target.is_symlink():
            target.unlink()
        print(f"Installing the published CoreX {version} wheels with Python 3.12", file=sys.stderr)
        print(version)
