"""BI-V150 reference selection for the legacy and IX RNG header layouts."""

import pytest

from benchmark.corex_matrix import corrected_build, sdk


@pytest.mark.parametrize("ix_headers", [False, True])
def test_corex_reference_selection(tmp_path, monkeypatch, ix_headers):
    root = tmp_path / "toolkit"
    include = root / "include"
    include.mkdir(parents=True)
    kernel = (
        "#include <IX/mapping_curand.h>\n\n#include <IX/curand_kernel.h>\n"
        if ix_headers
        else "struct curandStateMRG32k3a {};"
    )
    (include / "curand_kernel.h").write_text(kernel)
    monkeypatch.delenv("FLAGRAND_COREX_CURAND_PATH", raising=False)
    monkeypatch.delenv("CURAND_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("FLAGRAND_COREX_SDK_ROOT", str(root))
    distributions = ("raw32", "uniform_f32", "normal_f32", "lognormal_f32", "poisson")
    engines = ("philox", "xorwow", "mrg32k3a", "mtgp32")
    corrected = {(e, a) for e in engines for a in distributions if sdk.needs_corrected_reference(e, a)}
    expected = {("mrg32k3a", a) for a in distributions} | {(e, "poisson") for e in engines}
    assert corrected == (set() if ix_headers else expected)

    builds = []
    monkeypatch.setattr(corrected_build, "prepare_corrected_baseline", builds.append)
    monkeypatch.setattr(corrected_build, "baseline_build_record", lambda: {"status": "compiled"})
    out = tmp_path / "native-build"
    operators = [f"{engine}_{api}" for engine in engines for api in distributions]
    record = corrected_build.prepare_operator_references(operators, out)
    assert builds == ([] if ix_headers else [out])
    assert record == (None if ix_headers else {"status": "compiled"})


@pytest.mark.parametrize("variable", ["FLAGRAND_COREX_CURAND_PATH", "CURAND_LIBRARY_PATH"])
def test_explicit_stock_library_selects_its_own_headers(tmp_path, monkeypatch, variable):
    image_sdk = tmp_path / "corex44"
    active_sdk = tmp_path / "corex45"
    (active_sdk / "include").mkdir(parents=True)
    (active_sdk / "include/curand_kernel.h").write_text("#include <IX/curand_kernel.h>\n")
    (active_sdk / "lib64").mkdir()
    library = active_sdk / "lib64/libcurand.so"
    library.touch()
    monkeypatch.setenv("COREX_HOME", str(image_sdk))
    monkeypatch.delenv("FLAGRAND_COREX_CURAND_PATH", raising=False)
    monkeypatch.setenv(variable, str(library))
    assert sdk.sdk_root() == active_sdk.resolve()
    assert not sdk.needs_corrected_reference("mrg32k3a", "poisson")

    # A standalone library may have no headers alongside it.
    (active_sdk / "include/curand_kernel.h").unlink()
    monkeypatch.setenv("FLAGRAND_COREX_SDK_ROOT", str(image_sdk))
    assert sdk.sdk_root() == image_sdk.resolve()


def test_legacy_prebuilt_corrected_library_is_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAGRAND_COREX_CURAND_PATH", raising=False)
    monkeypatch.delenv("CURAND_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("FLAGRAND_COREX_SDK_ROOT", str(tmp_path))
    existing = tmp_path / "libflagrand_corex_corrected.so"
    existing.touch()
    record = tmp_path / "build.json"
    record.write_text('{"status":"compiled"}')
    monkeypatch.setenv(corrected_build.LIBRARY_ENV, str(existing))
    monkeypatch.setenv(corrected_build.RECORD_ENV, str(record))
    assert corrected_build.prepare_corrected_baseline() == str(existing)
    directory = tmp_path / "new-build"
    assert corrected_build.prepare_operator_references(["mtgp32_poisson"], directory) == {"status": "compiled"}
    assert not directory.exists()
