# FlagRAND

FlagRAND provides Triton host APIs for Philox4x32-10, XORWOW, MRG32K3A and MTGP32. The public surface includes raw32, float32/float64 uniform, normal and log-normal outputs, and integer Poisson outputs. The new generator families and Poisson paths are under device-specific performance iteration; see [RNG families](docs/usage/rng-families.md) for state semantics and current evidence.

## Layout

- `src/flagrand/ops/philox.py`: canonical operator facade.
- `src/flagrand/rng`, `fused`, `runtime`: shared implementation and backend dispatch.
- `src/flagrand/ascend`, `corex`, `metax`, `nvidia`: vendor contract adapters and kernels.
- `conf/operators.yaml`: operator registry and required test coverage.
- `tests/accuracy`: contract, deterministic and statistical correctness tests.
- `benchmark`: end-to-end NVIDIA, CoreX, PPU, MetaX/Muxi, Hygon and Ascend 910C performance matrices.
- `scripts`: reproducible platform and CI entry points.
- `docs`: installation, usage and contribution guides.

## Development checks

```bash
python -m pip install -e ".[dev]"
python tools/validate_operator_registry.py
pre-commit run --all-files
pytest -q tests/accuracy
python -m build
python -m twine check dist/*
```

## Documentation

- [Hardware acceptance](docs/getting-started/hardware-acceptance.md): clone, install,
  validate the device, run the acceptance matrix and verify `summary.json`.
- [NVIDIA H20](docs/usage/nvidia-h20.md): CUDA environment, native cuRAND comparison,
  correctness and performance matrices.
- [MetaX C550](docs/usage/non-nvidia/muxi.md): MACA container, native mcRAND comparison,
  platform dispatch and acceptance matrices.
- [C550 RNG families](docs/usage/c550-environment.md): the measured MACA image,
  source installation, four-generator reproduction and result conventions.
- [Hygon BW1000](docs/usage/non-nvidia/hygon.md): HIP/gfx936 dispatch, binary64
  parameter accuracy, native hipRAND comparison, unified summary.json and offline delivery.
- [Non-NVIDIA platforms](docs/usage/non-nvidia/): backend-specific runtime and
  diagnostic notes.
- [Development](docs/contribution/development.md): operator registration,
  correctness, performance and CI conventions.
