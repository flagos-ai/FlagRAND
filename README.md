# FlagRAND

FlagRAND provides a Philox4x32-10 host API implemented with Triton. The public surface mirrors the seven random-number operations required by the current delivery: raw32, uniform/normal/log-normal in float32 and float64-compatible forms.

## Layout

- `src/flagrand/ops/philox.py`: canonical operator facade.
- `src/flagrand/rng`, `fused`, `runtime`, `ascend`: implementation and backend support.
- `conf/operators.yaml`: operator registry and required test coverage.
- `tests/accuracy`: contract, deterministic and statistical correctness tests.
- `benchmark`: end-to-end NVIDIA, PPU, Muxi and Ascend 910C performance matrices.
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
- [Non-NVIDIA platforms](docs/usage/non-nvidia/): backend-specific runtime and
  diagnostic notes.
- [Development](docs/contribution/development.md): operator registration,
  correctness, performance and CI conventions.
