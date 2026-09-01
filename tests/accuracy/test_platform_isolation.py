from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_TRITON_SOURCES = (
    PROJECT_ROOT / "src/flagrand/rng/philox.py",
    PROJECT_ROOT / "src/flagrand/fused/_internal/philox_direct.py",
)
ASCEND_SOURCE = PROJECT_ROOT / "src/flagrand/ascend/huawei910c.py"
METAX_KERNEL_SOURCE = PROJECT_ROOT / "src/flagrand/metax/philox.py"
METAX_TRANSFORMS_SOURCE = PROJECT_ROOT / "src/flagrand/metax/transforms.py"
METAX_LAUNCHER_SOURCE = PROJECT_ROOT / "src/flagrand/metax/launcher.py"
SHARED_LAUNCHER_SOURCE = PROJECT_ROOT / "src/flagrand/runtime/compiled_launcher.py"
HOST_DISPATCH_SOURCE = PROJECT_ROOT / "src/flagrand/_curand_philox_fast_path.py"
ASCEND_OPS_SOURCE = PROJECT_ROOT / "src/flagrand/runtime/backend/_ascend/ops.py"
ILUVATAR_OPS_SOURCE = PROJECT_ROOT / "src/flagrand/runtime/backend/_iluvatar/ops.py"
METAX_OPS_SOURCE = PROJECT_ROOT / "src/flagrand/runtime/backend/_metax/ops.py"
NVIDIA_OPS_SOURCE = PROJECT_ROOT / "src/flagrand/runtime/backend/_nvidia/ops.py"


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


def test_metax_implementation_is_independent_of_shared_rng_kernels() -> None:
    forbidden = {
        "flagrand.rng.philox",
        "flagrand.fused._internal.philox_direct",
        "flagrand.fused._internal.philox_direct_f64",
        "flagrand.fused._internal.transforms",
        "flagrand.runtime.compiled_launcher",
    }
    for path in (METAX_KERNEL_SOURCE, METAX_TRANSFORMS_SOURCE, METAX_LAUNCHER_SOURCE):
        assert _imported_modules(path).isdisjoint(forbidden)
    assert "tl.philox" in METAX_KERNEL_SOURCE.read_text()
    assert "class MetaXKernelLauncher" in METAX_LAUNCHER_SOURCE.read_text()
    assert "maca" not in SHARED_LAUNCHER_SOURCE.read_text().lower()


def test_host_dispatch_has_no_eager_ascend_import() -> None:
    tree = ast.parse(
        HOST_DISPATCH_SOURCE.read_text(),
        filename=str(HOST_DISPATCH_SOURCE),
    )
    top_level_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("flagrand.ascend") for module in top_level_imports)


def test_generic_dispatch_has_no_vendor_branches() -> None:
    source = HOST_DISPATCH_SOURCE.read_text()
    imports = _imported_modules(HOST_DISPATCH_SOURCE)
    assert "corex" not in source.lower()
    assert "ascend" not in source.lower()
    assert "metax" not in source.lower()
    assert not any(module.startswith("flagrand.corex") for module in imports)
    assert not any(module.startswith("flagrand.ascend") for module in imports)
    assert not any(module.startswith("flagrand.metax") for module in imports)


def test_specialized_ops_live_under_vendor_backends() -> None:
    assert any(module.startswith("flagrand.ascend") for module in _imported_modules(ASCEND_OPS_SOURCE))
    assert any(module.startswith("flagrand.corex") for module in _imported_modules(ILUVATAR_OPS_SOURCE))
    assert any(module.startswith("flagrand.metax") for module in _imported_modules(METAX_OPS_SOURCE))
    assert any(module.startswith("flagrand.nvidia") for module in _imported_modules(NVIDIA_OPS_SOURCE))


def test_nvidia_backend_owns_the_complete_public_dispatch() -> None:
    tree = ast.parse(NVIDIA_OPS_SOURCE.read_text(), filename=str(NVIDIA_OPS_SOURCE))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "generate",
        "generate_uniform",
        "generate_uniform_double",
        "generate_normal",
        "generate_normal_double",
        "generate_lognormal",
        "generate_lognormal_double",
    }.issubset(functions)


def test_metax_backend_owns_the_complete_public_dispatch() -> None:
    tree = ast.parse(METAX_OPS_SOURCE.read_text(), filename=str(METAX_OPS_SOURCE))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "generate",
        "generate_uniform",
        "generate_uniform_double",
        "generate_normal",
        "generate_normal_double",
        "generate_lognormal",
        "generate_lognormal_double",
    }.issubset(functions)
