# FlagRAND

FlagRAND provides a Philox4x32-10 host API implemented with Triton. The public surface mirrors the seven random-number operations required by the current delivery: raw32, uniform/normal/log-normal in float32 and float64-compatible forms.

## Layout

- `src/flagrand/ops/philox.py`: canonical operator facade.
- `src/flagrand/rng`, `fused`, `runtime`, `ascend`: implementation and backend support.
- `conf/operators.yaml`: operator registry and required test coverage.
- `tests/accuracy`: contract, deterministic and statistical correctness tests.
- `benchmark`: end-to-end PPU, Muxi and Ascend 910C performance matrices.
- `scripts`: reproducible platform and CI entry points.
- `docs`: platform guides and validated reports.

## Development checks

```bash
python -m pip install -e ".[dev]"
python tools/validate_operator_registry.py
pre-commit run --all-files
pytest -q tests/accuracy
python -m build
python -m twine check dist/*
```

Hardware execution must use the vendor environment already installed on the runner; do not replace its torch, Triton or device runtime packages. See [PPU](docs/platforms/ppu.md) and [Ascend 910C](docs/platforms/ascend_910c.md) for exact commands.

The delivery one-click entry points are `scripts/delivery/run_ppu_v6.sh` and
`scripts/delivery/run_ascend_910c_v6.sh`. The Ascend acceptance entry point runs
each public API in an isolated process and applies the `0.95x` gate per API.
