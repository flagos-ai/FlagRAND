from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_TRITON_SOURCES = (
    PROJECT_ROOT / "src/flagrand/rng/philox.py",
    PROJECT_ROOT / "src/flagrand/fused/_internal/philox_direct.py",
)
ASCEND_SOURCE = PROJECT_ROOT / "src/flagrand/ascend/huawei910c.py"
HOST_DISPATCH_SOURCE = PROJECT_ROOT / "src/flagrand/_curand_philox_fast_path.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_ppu_cuda_kernels_are_ascend_independent() -> None:
    for path in SHARED_TRITON_SOURCES:
        source = path.read_text()
        imports = _imported_modules(path)
        assert '"npu"' not in source
        assert "philox_core" not in source
        assert not any(module.startswith("flagrand.ascend") for module in imports)
        assert "tl.philox" in source


def test_ascend_kernel_does_not_import_ppu_cuda_kernels() -> None:
    imports = _imported_modules(ASCEND_SOURCE)
    forbidden = {
        "flagrand.rng.philox",
        "flagrand.fused._internal.philox_direct",
        "flagrand.fused._internal.philox_direct_f64",
    }
    assert imports.isdisjoint(forbidden)


def test_host_dispatch_has_no_eager_ascend_import() -> None:
    tree = ast.parse(
        HOST_DISPATCH_SOURCE.read_text(),
        filename=str(HOST_DISPATCH_SOURCE),
    )
    top_level_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("flagrand.ascend") for module in top_level_imports)
