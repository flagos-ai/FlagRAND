from __future__ import annotations

import ctypes
import sys

import pytest

from benchmark.muxi_matrix.config import PROFILES, parse_args
from benchmark.muxi_matrix.mcrand import McrandError, _bind_signatures, _check, mcrand_library_path


class _Function:
    argtypes = None
    restype = None


class _Library:
    def __init__(self) -> None:
        for name in (
            "mcrandCreateGenerator",
            "mcrandDestroyGenerator",
            "mcrandGetVersion",
            "mcrandSetPseudoRandomGeneratorSeed",
            "mcrandSetGeneratorOffset",
            "mcrandSetStream",
            "mcrandGenerate",
            "mcrandGenerateUniform",
            "mcrandGenerateUniformDouble",
            "mcrandGenerateNormal",
            "mcrandGenerateNormalDouble",
            "mcrandGenerateLogNormal",
            "mcrandGenerateLogNormalDouble",
        ):
            setattr(self, name, _Function())


def test_mcrand_library_path_prefers_explicit_environment(monkeypatch, tmp_path) -> None:
    library = tmp_path / "libmcrand.so"
    library.touch()
    monkeypatch.setenv("FLAGRAND_MCRAND_PATH", str(library))

    assert mcrand_library_path() == str(library)


def test_mcrand_ctypes_signatures_cover_all_delivery_apis() -> None:
    library = _Library()

    _bind_signatures(library)

    assert library.mcrandCreateGenerator.argtypes == [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    assert library.mcrandGetVersion.argtypes == [ctypes.POINTER(ctypes.c_int)]
    assert library.mcrandGenerate.argtypes == [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    assert library.mcrandGenerateUniformDouble.restype is ctypes.c_int
    assert library.mcrandGenerateNormal.argtypes[-2:] == [ctypes.c_float, ctypes.c_float]
    assert library.mcrandGenerateNormalDouble.argtypes[-2:] == [ctypes.c_double, ctypes.c_double]
    assert library.mcrandGenerateLogNormal.argtypes[-2:] == [ctypes.c_float, ctypes.c_float]
    assert library.mcrandGenerateLogNormalDouble.argtypes[-2:] == [ctypes.c_double, ctypes.c_double]


def test_mcrand_status_names_are_reported() -> None:
    with pytest.raises(McrandError, match="MCRAND_STATUS_LENGTH_NOT_MULTIPLE"):
        _check(105, "mcrandGenerateNormal")


def test_muxi_defaults_to_mcrand_baseline(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark.run_muxi", "--skip-torch-reference"])

    args = parse_args(tmp_path)

    assert args.acceptance_baseline == "mcrand"
    assert not args.skip_mcrand
    assert args.skip_ppu_curand
    assert args.skip_ppu_acrand


def test_ppu_defaults_keep_mcrand_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark.run_ppu", "--skip-torch-reference"])

    args = parse_args(
        tmp_path,
        default_results_dir_name="ppu_results",
        default_skip_mcrand=True,
        default_skip_ppu_curand=False,
        default_skip_ppu_acrand=False,
        default_acceptance_baseline="ppu_acrand",
        platform_name="PPU",
    )

    assert args.acceptance_baseline == "ppu_acrand"
    assert args.skip_mcrand
    assert not args.skip_ppu_curand
    assert not args.skip_ppu_acrand


def test_muxi_smoke_prewarms_all_stateful_philox_specializations() -> None:
    assert PROFILES["smoke"].warmup == 8
