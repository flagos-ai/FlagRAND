# Hygon BW1000 / gfx936

The Hygon adapter has been exercised on a BW1000 node using PyTorch 2.10.0
(`2.10.0+das.opt1.dtk2604.2606151054.g98b1a8`) and its installed Triton 3.6.0 HIP
backend, reporting `gfx936` with a 64-thread warp. This is the validated
environment; other Hygon architectures and SDK releases need their own runs.

## Running from an offline source copy

Use the node's existing matching PyTorch/Triton environment. No GitHub connection
is needed at runtime:

```bash
export PYTHONPATH=/path/to/FlagRAND/src
python -c 'import flagrand; from flagrand.runtime import current_vendor; print(current_vendor())'
```

Detection recognizes HIP `gfx936`, the `hcu` target, and the `hy-smi` system
query. An explicit `FLAGRAND_VENDOR=hygon` also selects this adapter. GPU tensors
use PyTorch's `cuda` device namespace.

## Semantics and implementation

- All seven public Philox APIs are implemented. Raw words use the shared
  Philox4x32-10 mapping with counter words `(0, 0, counter_low, counter_high)`.
- FP32 normal and lognormal use portable math, with no NVIDIA inline PTX.
- FP64 normal and lognormal preserve the binary64 representation of `mean`
  and `stddev` by passing their bits as integer launch arguments. HIP libdevice
  supplies double-precision log, sqrt, and exp. The existing double-precision
  sine/cosine polynomial is reused.
- Counter additions promote to uint64 before adding vector offsets, including
  when a Triton argument is specialized to a Python integer constant.
- The standard compiled-kernel launch cache is enabled for the tested SDK.
  `FLAGRAND_FORCE_PORTABLE=1` uses regular JIT launching for comparison.

The mapping follows the documented FlagRAND portable contract. Native hipRAND
sequence equivalence has **not** been measured or asserted.

## Validation and measured changes

The targeted public API, backend, stream, deterministic, statistical, and Hygon
regressions passed **100 tests**, with **8 CoreX-specific tests skipped**:

```bash
pytest -q tests/accuracy/test_api.py tests/accuracy/test_backend.py \
  tests/accuracy/test_backend_registry.py tests/accuracy/test_philox_cuda.py \
  tests/accuracy/test_hygon.py
```

An independent CPU Philox/transform oracle also compared seven distributions,
two initial shapes/offsets, and six launch settings. At `n=4096/4100`,
`offset=0/3`, `seed=20260914`, `mean=0.2`, `stddev=0.7`:

| Measurement | Before | After |
|---|---:|---:|
| FP64 normal maximum absolute error | about 5.1e-8 | below 9e-16 |
| FP64 lognormal maximum absolute error | about 6.7e-7 | below 9e-15 |
| Raw32 complete API call, n=4100, block=512 | 69.6 us, regular JIT | 31.6 us, cached launch |
| Same raw32 call inside a captured graph | about 4.88 us | about 4.88 us |

The launch change saves host-side work. It does not make the raw32 kernel itself
twice as fast. Block-size tuning is separate: at that same unaligned input,
block 256 measured about 4.04 us inside a graph, versus 7.10 us for block 1024.
These are sampled input points, not a general performance guarantee.
The graph benchmark fixes seed/offset; advancing generator state on successive
graph replays is not implemented or tested by this measurement.

Operator Atlas retains the complete samples, errors, compiler metadata, and
source hashes. Source branch: `codex/atlas-hygon`, based on upstream `88dd5ce`.

## Complete offline delivery

Use the existing DTK PyTorch/Triton environment; the delivery entry does not
install packages, download code, or access GitHub. It needs Python 3.10+, Torch,
the matching Hygon Triton, NumPy and pytest. The tested native library is DTK
`/opt/dtk/lib/libhiprand.so` (hipRAND version integer 300200). Set
`FLAGRAND_HIPRAND_PATH` if its location differs.

From a complete source copy (including `benchmark`, `tools`, `conf`, `scripts`
and `tests`):

```bash
export PYTHON_BIN=/usr/bin/python3
export CUDA_VISIBLE_DEVICES=1
export FLAGRAND_RUN_ID=hygon_acceptance
bash scripts/delivery/run_hygon.sh
```

This defaults to the full `acceptance` profile. For a short end-to-end run:

```bash
FLAGRAND_PERF_PROFILE=smoke FLAGRAND_RUN_ID=hygon_smoke bash scripts/delivery/run_hygon.sh
```

Both commands invoke `scripts/ci/run_backend_ci.sh hygon`. The same unified
runner is also available directly:

```bash
source scripts/ci/backend_env.sh hygon
FLAGRAND_PLATFORM=hygon FLAGRAND_PERF_PROFILE=acceptance \
  "$PYTHON_BIN" tools/run_tests.py --gpus 0 --stages stable \
  --output-dir hygon_results/manual --dump-output
```

The selected GPU remains masked in child processes. When run inside Operator
Atlas, its agent also recognizes benchmark descendants as our workload and
releases the entire worker container when another user needs the node.

## Delivery artifacts and comparison contract

- `hygon_results/ci_<RUN_ID>/summary.json`: the same unified delivery schema as
  other chips (`timestamp`, `total_duration`, `env`, `result`). The five delivery
  names are `PHILOX4_32_10`, `RAW_BITS`, `UNIFORM`, `NORMAL`, and `LOGNORMAL`.
- Each API folder contains accuracy/performance result JSON and captured logs.
  Its `performance_raw/` contains `manifest.json`, `records.jsonl`, `records.csv`,
  `summary.json`, and `summary.md` from the actual matrix.
- `ci_logs/hygon/<RUN_ID>/` contains the environment, common runner regression,
  Hygon regressions and complete runner log.

The full matrix uses the same shapes as the seven-API NVIDIA matrix: 44 raw/
uniform shapes and 42 normal/lognormal shapes, totaling 300 API/shape pairs.
Tiny, tail and power-of-two sizes are included, up to 16,777,216 elements.
Each pair measures FlagRAND and native hipRAND. Both reuse a persistent generator,
advance RNG state, write preallocated output and use the current HIP stream.
Compilation/warmup are excluded. The formal metric is synchronized end-to-end
`total_ms_median`; enqueue/device metrics and raw timing samples are retained. HIP events span the
submitted call batch and may include gaps while Python submits work; the event
metric is not an isolated kernel latency or a vendor profiler breakdown.
The per-API score is the arithmetic mean of each shape's native/FlagRAND ratio.
There is no cross-API mean and no invented speedup cutoff.

FlagRAND is checked against independent CPU Philox arithmetic, sequence offsets,
replay, streams, finite/range and statistical tests. Native hipRAND is checked
for its own replay, stream and distribution behavior. The two libraries need
not return the same ordered sequence for the same seed. This is recorded in
manifests rather than hidden behind an elementwise native-equality assertion.

For a single benchmark or compiler/reference probe:

```bash
python3 -m benchmark.run_hygon --profile matrix --distributions raw32 --results-dir hygon_results/raw
python3 -m benchmark.run_hygon --preflight-only
```

To install a development environment on a connected runner, `./setup.sh hygon`
uses the existing system Torch/Triton. GitHub Actions supports the `hygon` backend
and includes `hygon_results/` in result uploads; registering the actual
self-hosted runner is repository infrastructure work and is not done by these scripts.

## Full delivery measured on 2026-09-14

Run `692cd236f0bd470da272ed60bce6c097` completed all 14 accuracy/performance
phases with zero failures. The 300 API/shape pairs produced 600 passing records.
The per-API accuracy reports contain 111 unique passed tests and eight skipped
CoreX-only tests; separate CI stages passed 27 Hygon/API/backend tests and 14
runner tests. The unified runner took 11 minutes 10 seconds.

| API | Shapes | Mean hipRAND / FlagRAND | At 1,048,576 | At 16,777,216 |
|---|---:|---:|---:|---:|
| raw32 | 44 | 0.860x | 1.062x | 1.625x |
| uniform_f32 | 44 | 0.805x | 0.987x | 1.471x |
| uniform_f64 | 44 | 0.601x | 1.018x | 1.122x |
| normal_f32 | 42 | 0.722x | 2.000x | 0.879x |
| normal_f64 | 42 | 0.540x | 0.608x | 0.495x |
| lognormal_f32 | 42 | 0.633x | 0.613x | 1.110x |
| lognormal_f64 | 42 | 0.545x | 0.601x | 0.505x |

All full-matrix API means are below 1. Passing delivery means the tests and
benchmark completed correctly; it does not mean every shape beats hipRAND.
The main remaining work is small-input host/launch cost and FP64 distribution
transforms. FP32 normal changes from 1.560x at 4M to 0.879x at 16M; FP32
lognormal changes from 0.613x at 1M to 1.150x at 4M. These sampled transitions
are evidence for further tuning, not exact dispatch thresholds. Near-1 ratios
must not be treated as reliable wins from a single matrix run.

Validated runtime source SHA256:
`b7228d5dc516a5b7c0b99be4b64f8e51e7f885a7373ada8a91fbe28241122917`.
Validated benchmark/tools SHA256:
`88586089d25379e4b6cde7fb12976a0e3ffa3b8e36378de31f09e7a384940226`.


## Baseline optimization, 2026-09-14

The follow-up complete matrix meets the same per-API arithmetic-mean target
(`>= 0.95x`) documented for MetaX C550 and Ascend. On BW1000 / gfx936,
DTK 26.04, Torch 2.10.0 and Triton 3.6.0, all 300 API/shape pairs and
600 implementation/baseline records passed. All 14 CI phases passed;
independent prerequisite suites passed 27 Hygon/API tests and 14 runner tests.

| API | 原始交付 | 优化后 | ≥0.95× |
|---|---:|---:|---|
| raw32 | 0.860× | 1.072× | 达标 |
| uniform_f32 | 0.805× | 1.517× | 达标 |
| uniform_f64 | 0.601× | 1.290× | 达标 |
| normal_f32 | 0.722× | 1.138× | 达标 |
| normal_f64 | 0.540× | 1.008× | 达标 |
| lognormal_f32 | 0.633× | 0.984× | 达标 |
| lognormal_f64 | 0.545× | 1.001× | 达标 |

The ratio is native hipRAND / FlagRAND synchronized host API time; larger
means faster. These are API means, not a guarantee for every input.
For example, raw32 at N=1,048,575 is 0.471x, normal_f64 has a 0.623x slow
point, and normal_f32 at 16M is 0.857x. In normal_f32, 36 of 42 shapes remain
below 1.0x despite its 1.138x API mean. Retain the per-shape records.

Implementation follows NVIDIA's direct public API dispatch structure while
preserving Hygon's portable Philox stream and two uint32 words per FP64 output.
FP64 launch keys now distinguish device, pointer alignment, seed, counter
specialization class, size, SHIFT, parameters and block configuration. Every
call resolves the current stream. Standard normal inputs omit affine work;
nonstandard binary64 mean/stddev retain exact bit transport.

The selected defaults are BLOCK=256/warps=4 for raw/uniform, 128/2 for FP32
normal/lognormal, and 256/4 for FP64 normal/lognormal. Explicit launch options
remain available. A repeated portability check on modern Triton was removed
by checking the legacy-version condition first; dynamic portable mode and
profiling hooks still take their existing paths.

The old FP64 128/4 configuration generated a layout conversion and 2048 bytes
of launch-time LDS: `ds_write2st64_b64`, `s_barrier`, `ds_read_b64`, then an
8-byte output store. The 256/4 configuration uses no LDS or conversion and a
16-byte output store. For normal_f64, graph timing fell from 43.32 to 21.47 us
at 1M and 649.89 to 305.17 us at 16M. VGPR usage was 32 versus 31 with no
spills. Assembly fixed-LDS declarations alone miss the dynamic launch allocation.

A separate FP32 crossover remains unexplained: 128/2 versus 256/4 at 1M is
12.25 versus 55.18 us, despite identical VGPR count and store opcode; at 16M
it is 152.38 versus 148.79 us. This supports configuration selection, but is
not sufficient evidence to claim a specific cache-conflict mechanism.

`python benchmark/run_hygon_tuning.py --output hygon_results/tuning` runs the
finite host profiles and configuration experiments. Use the normal delivery
script with `FLAGRAND_PERF_PROFILE=acceptance` for formal measurements.
This run ID is `4abbd70eb84941879227f0c0c428e351` on physical GPU 2 of the
validated node. The original baseline data above remains a historical snapshot.

Chinese delivery documents: [Hygon_BW1000_FlagRAND_性能正确性与实现说明](./Hygon_BW1000_FlagRAND_性能正确性与实现说明.md) and [Hygon_BW1000_FlagRAND_正确性验证策略](./Hygon_BW1000_FlagRAND_正确性验证策略.md).
