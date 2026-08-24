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

## Unified hardware acceptance

Clone the delivery branch over SSH and enter the repository:

```bash
git clone --branch main --single-branch git@github.com:flagos-ai/FlagRAND.git
cd FlagRAND
```

The unified runner uses the backend definitions in `conf/backends.yaml`, writes a
FlagGems-compatible `summary.json`, and exposes exactly five delivery results:
`PHILOX4_32_10`, `RAW_BITS`, `UNIFORM`, `NORMAL`, and `LOGNORMAL`.

### Iluvatar BI-V150 / CoreX 4.4

```bash
./setup.sh iluvatar
source scripts/ci/backend_env.sh iluvatar
bash tools/gpu_check_iluvatar.sh

RUN_ID="corex_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_RUN_ID="$RUN_ID" \
FLAGRAND_PERF_PROFILE=acceptance \
bash scripts/ci/run_backend_ci.sh iluvatar

.venv/bin/python -m json.tool "corex_results/ci_${RUN_ID}/summary.json"
```

### PPU-ZW810E

```bash
./setup.sh thead
source scripts/ci/backend_env.sh thead
bash tools/gpu_check_thead.sh

RUN_ID="ppu_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_RUN_ID="$RUN_ID" \
FLAGRAND_PERF_PROFILE=acceptance \
bash scripts/ci/run_backend_ci.sh thead

.venv/bin/python -m json.tool "ppu_results/ci_${RUN_ID}/summary.json"
```

### Ascend / CANN 9.0

```bash
./setup.sh ascend-cann900
source scripts/ci/backend_env.sh ascend-cann900
bash tools/gpu_check_ascend.sh

RUN_ID="ascend_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_RUN_ID="$RUN_ID" \
FLAGRAND_PERF_PROFILE=acceptance \
bash scripts/ci/run_backend_ci.sh ascend-cann900

.venv/bin/python -m json.tool "ascend_910c_results/ci_${RUN_ID}/summary.json"
```

The same jobs can be dispatched from GitHub Actions through `unit-test` with
backend `iluvatar`, `thead`, or `ascend-cann900` and profile `acceptance`.

PPU execution uses the vendor environment already installed on the runner. A fresh Ascend 910C node with the NPU driver, Docker and Ascend Docker Runtime can build the pinned CANN 9.0 environment and run the full acceptance workflow with:

```bash
bash run_ascend_910c_acceptance.sh
```

See [BI-V150/CoreX](docs/platforms/corex_biv150.md), [PPU](docs/platforms/ppu.md), and [Ascend 910C from-scratch delivery](docs/platforms/ascend_910c_environment.md) for the exact environment contracts.

The delivery one-click entry points are `scripts/delivery/run_ppu_v6.sh` and
`scripts/delivery/run_ascend_910c_v6.sh`. The Ascend acceptance entry point runs
each public API in an isolated process and applies the `0.95x` gate per API.
