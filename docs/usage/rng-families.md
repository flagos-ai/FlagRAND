# Random-number generator families

The host API uses cuRAND's operation names as a functional reference. Seeds
need not produce the same bits as cuRAND. Generator recurrences, state
continuation, repeatability and distribution mathematics are tested separately.

```python
import torch
from flagrand import curand

generator = curand.create_generator("xorwow", seed=20261020)
counts = torch.empty(65536, dtype=torch.int32, device="cuda")
curand.generate_poisson(generator, counts, lam=10.0)
curand.set_pseudo_random_generator_seed(generator, 20261020)
curand.generate_poisson(generator, counts, lam=10.0)  # repeat the first call
```

Available names are `philox`, `xorwow`, `mrg32k3a` and `mtgp32`, with exported
`CURAND_RNG_PSEUDO_*` constants. Each supports `generate`,
`generate_uniform[_double]`, `generate_normal[_double]`,
`generate_lognormal[_double]` and `generate_poisson`. Raw uint32 words use
`torch.int32` storage; interpret their bits as unsigned. Poisson also uses
integer storage. Its scalar lambda parameter does not define a float output
precision. Normal and log-normal accept `mean` and `stddev`; for log-normal,
these are the parameters of the underlying normal variable.

## State and sequence conventions

* Philox retains the existing scalar raw-word offset semantics.
* XORWOW reuses the six-word persistent streams and backend implementations
  from the project's existing `xorwow_kernels`. The seed mapping uses
  SplitMix64. A fixed 32768 streams are interleaved in each call. Resetting a
  seed and replaying the same call sequence reproduces the result. Concatenated
  calls match one call only at complete stream rows (two rows for paired
  normal outputs); arbitrary partial calls restart interleaving at stream zero.
  This is not cuRAND's `2^67` jump initialization.
  On MetaX, repeated calls reuse compiled kernels and cache the resolved
  backend and base launch profile by requested backend, device, shape,
  distribution and dtype. Explicit options update a fresh configuration copy.
  The current stream, output address, `mean` and `stddev` remain per-call
  values. These host caches preserve the distribution mathematics and fixed
  32768 streams.
  Large MetaX binary64 normal and log-normal calls separate integer generation
  from the distribution transform. The second kernel processes all output
  pairs independently while preserving the original draw-major layout,
  six-word state and FP64 math. This path requires at least 1048576 outputs,
  complete paired stream rows, and the default MetaX math configuration;
  partial rows and explicit kernel options retain the fused path.
  Each active stream retains an int32 workspace sized for twice its largest
  qualifying output count: 128 MiB for 16777216 outputs, in addition to the
  output and state. Seed reset releases the generator's references to these
  buffers; the Torch allocator manages cached device memory. Switching streams with
  one generator requires the caller to order access to its shared state.
  The steady-state benchmark warms up allocation and compilation; a cold
  call or a larger requested capacity also pays for workspace allocation.
* MRG32K3A uses the two third-order modular recurrences and `2^76` stream
  spacing. The default 131072 streams and a persistent cursor define output
  ordering. Same-distribution calls preserve ordering across partial shapes.
  The earlier 65536-stream layout remains selectable with `nstreams=65536`;
  changing this count changes the interleaved output sequence. The new default
  stores 3 MiB of generator state. Measured initialization (CPU construction,
  transfer and synchronization) rose from about 0.42 to 0.85 seconds on C550,
  outside the steady-state timing interval.
  Raw values occupy `[1,4294967087]`, rather than all uint32 values. Double
  uniform combines two base-modulus digits instead of casting a float32 value.
  Its default reciprocal multiplication preserves the integer sequence but can
  change final floating-point bits relative to the division variant. The
  measured comparison reached a distance of two binary64 representable values;
  this is a sampled error result, not an all-input correct-rounding claim.
  Normal and log-normal use both outputs of Box–Muller. Each precision keeps
  one pending standard normal after an odd tail; other distributions and the
  other precision leave it intact. The next normal or log-normal call applies
  its current parameters to that pending value. Include these pending values
  and their flags when reasoning about continuation.
* MTGP32 reuses the 192 parameter sets and 256-word update order in the existing
  implementation. Unconsumed words in a partial update remain in a raw cache,
  including when the next call changes distribution. Arbitrary offset/skipahead
  is unsupported, consistent with cuRAND's host MTGP32 restriction.
  Partial MetaX calls prefetch approximately 32 calls' raw-word consumption,
  capped at 1,048,576 words (4 MiB), rounded to complete 256-word updates.
  Short MetaX raw32 calls at `N <= 65536`, including fresh aligned calls,
  use this existing raw window. Cache hits copy the requested words through
  MACA D2D on the current stream; the existing refill path that also writes
  public raw output remains in use. This adds no derived array and leaves
  the total derived-storage bound below unchanged. Raw order, consumed
  offset and mixed-distribution continuation are preserved; explicit kernel
  options retain the existing path.
  Cache-only refills use a persistent 1024-word shared state ring when at least
  8192 new 256-word blocks are needed. Smaller refills and raw/FP32-uniform
  refills that also write the public output retain the existing global path.
  Requests above that cap generate only the current request and its partial
  final update. Distribution conversion reads the retained raw cache directly.
  The physical `engine.state` may therefore lead the consumed `engine.offset`.
  A restorable continuation includes the state tensor, seed, consumed offset,
  physical frontier, raw cache and cache start; the state tensor alone is not
  sufficient. An aligned call consumes any remaining cache before using an
  aligned generation path. Explicit launch options retain the existing path.
  The subsequent r6 implementation sends short MetaX uniform FP64 calls
  (`N <= 65536`), including fresh aligned calls, through this bounded raw
  window. For each raw tensor, it lazily converts each offset parity once
  with the existing 53-bit mapping. The two parity arrays retain at most
  `8 * (1048576 + 255)` bytes for the active raw window. Raw words remain
  part of continuation state; these derived FP64 values can be rebuilt.
  Exhausting or replacing the raw cache, or resetting the generator, clears
  the derived references. Stream-ordered allocator reuse can outlive those
  Python references until queued work completes.
  The r7 integration extends the same short-input path to uniform FP32 at
  `N <= 65536`. One FP32 array is derived from the active raw window, retaining
  at most `4 * (1048576 + 255)` bytes. Mixed FP32/FP64 calls therefore retain
  at most `12 * (1048576 + 255) = 12585972` bytes (about 12 MiB) of derived
  tensors, excluding the raw window itself. Both caches are released on raw
  window exhaustion, replacement or seed reset; larger calls keep the
  existing generation path and mapping.
  Short MetaX Poisson calls with `N <= 65536` also reuse derived values from
  the bounded raw window. The generator retains only the latest table
  identity: raw storage, threshold and alias tensors, first value, and table
  size. At most two int32 arrays represent the two raw-offset parities.
  Changing the table replaces those arrays instead of keeping a per-lambda
  map. This adds at most `4 * (1048576 + 255)` bytes. Uniform FP32, uniform
  FP64 and Poisson together retain at most
  `16 * (1048576 + 255) = 16781296` bytes (about 16 MiB) of derived tensors,
  excluding raw storage and the existing alias tables. Oversized raw windows
  do not create derived values. Raw-window replacement or exhaustion and
  seed reset clear the derived references; copies and storage lifetime follow
  the current stream.
  Short MetaX normal and log-normal calls in both precisions also use this
  path at `N <= 65536`. The four APIs share one recent `_normal_cache` entry:
  raw tensor, mode, dtype, mean/stddev bits, raw-offset residue, derived values
  and stream. Each output pair consumes two raw words for FP32 or four for
  FP64. A change in mode, dtype, parameters or residue replaces the one
  derived array instead of accumulating parameter entries. Its storage is
  bounded by `4 * (1048576 + 255)` bytes, about 4 MiB. Together with uniform
  and Poisson, active derived values retain at most
  `20 * (1048576 + 255) = 20976620` bytes, about 20 MiB. Raw storage, existing
  alias tables and old storage retained by pending stream work are separate.
  Oversized raw windows do not create derived values; raw-window exhaustion,
  replacement and seed reset clear the references. Stream changes preserve
  queued storage lifetime and use the caller's explicitly ordered state.
  Output copying uses MACA `mcMemcpyAsync` for device-to-device transfer on
  the current stream. FlagRAND still generates the random values; mcRAND
  generation is not used by this path. The first runtime-library binding and
  first derived conversion are covered by warmup; subsequent refills,
  conversion rebuilds and every copy remain inside full-API timing.
  The independent r6 run completed all 520 paired points and 111 accuracy
  executions, including 17 uniform FP64 tests. Its two cache-lifetime seed
  cases each checked 48 mixed calls and 432 comparisons. All eight MTGP32
  APIs reached a 0.95 geometric mean; uniform FP64 was 1.546692.
  This is a separate eight-API run, not a replacement of the r5 snapshot.
  The r7 production integration was measured separately on physical GPU 6:
  all 44 uniform FP32 points completed, with a 1.151110 geometric mean and
  14 accuracy executions passing. Its preceding uniform FP64 marker run
  passed 17 tests. Each of the 88 paired-side records retains 11 timing
  samples. The 35 points at `N <= 65536` had a 1.273707 geometric mean;
  the other nine points were 0.776587 and remain in the full result.
  The subsequent r8 integration completed all 220 MTGP32 Poisson shape/lambda
  points on physical GPU 3, retaining 440 paired-side timing records. Its
  geometric mean against the matching mcRAND API was 1.488941, and its
  arithmetic mean was 1.558149. All 16 accuracy executions passed, including
  two new cache-lifetime seed cases with 31 observations and 220 Boolean
  comparisons each. The geometric means of FlagRAND/mcRAND API times were
  13.733/20.448 microseconds across all 220 points. The 175 short points were
  9.918/17.109 microseconds (1.725027), while the other 45 points were
  48.693/40.905 microseconds (0.840051). All points remain in the result,
  with 11 timing samples in each raw record.
  The r9 production integration on physical GPU 3 completed 42 points for
  each normal/log-normal API, 168 paired points in total. The geometric means
  were 1.522629 for normal FP32, 1.431151 for normal FP64, 1.024151 for
  log-normal FP32 and 1.614236 for log-normal FP64. The four API markers
  passed 14, 15, 14 and 19 accuracy executions respectively: 62 passed,
  zero failed and zero skipped, with no separate preceding tests. Eight
  cache-lifetime cases (two seeds per API) completed 352 observations and
  2984 Boolean comparisons, all true. All 168 points and 336 raw records
  remain in the result; independent implementation comparisons are separate.
  These are separate API integrations, not a complete 32-API fresh-clone
  result. The first published snapshot's fresh-clone run measured uniform
  FP32 at 0.745122. The later `9554017` snapshot's fresh-clone r2 measured
  all 220 MTGP32 Poisson points at 0.705456, below the 0.95 target.
  The subsequent `6150780` snapshot's fresh-clone r3 measured all 42 MTGP32
  log-normal FP64 points at 0.873118, also below target. The `22ebe335`
  snapshot's fresh-clone r4 measured all 44 MTGP32 raw32 points at 0.773868.
  Its relative change also involves lower native-library timing and does
  not establish a FlagRAND code regression. The new raw cache-hit D2D path
  completed an independent r10 production run on physical GPU 5 over all
  44 raw32 points and 88 raw timing records. All 17 accuracy executions
  passed, with zero failed and zero skipped. Its geometric mean was 1.248531
  and arithmetic mean was 1.278275. FlagRAND/mcRAND full-API time geometric
  means were 7.877/10.992 microseconds over the 35 short points (1.395348),
  and 26.360/21.358 over the remaining nine points (0.810272). The eight
  points below 0.95 remain in the result. Its preceding control/candidate
  experiment showed host-API gains while device activity increased; that
  experiment was not a native-library comparison. The r10 production run
  covers raw32 alone and does not replace r4 data.
  The revised source still needs its own complete 32-API run; local repairs
  do not replace individual APIs inside earlier runs.
  With the MetaX FlagTree TLE interface, aligned generation of at least
  1,048,576 raw words uses a persistent 1024-word shared array per state.
  This selection requires four warps and the linear parameter tables. It
  preserves the 192-state order, all physical state words and raw continuation;
  smaller calls, other launch configurations and older backends retain the
  global-state kernel. Compilation and keyed-launch options still apply to
  the selected kernel. The shared path also supplies raw words to separate
  normal and log-normal transforms. Partial-cache refills use the
  separate selection described above.
  Aligned uniform FP64 and Poisson use a corresponding shared-state pair
  kernel when TLE is available, two warps are requested and the call consumes
  at least 1,048,576 raw words (524,288 outputs). Each lane retains ownership
  of two adjacent words; parameter tables, alias sampling and continuation
  remain unchanged. Smaller calls and other configurations use the existing
  direct-pair kernel. Poisson tables with at most 1024 entries are copied once
  per CTA into shared memory, adding at most 8 KiB; larger tables retain the
  original global lookup. The alias thresholds and raw-word mapping are unchanged.
  The original-trigonometry FP32 normal/log-normal direct-pair path also uses
  a separate shared-state entry when TLE is available, two warps are requested,
  at least 1,048,576 raw words are consumed and at most 1,048,576 outputs are
  produced. The state order, math, output order and current-stream handling
  remain unchanged. Smaller calls retain the global-state pair kernel;
  larger calls retain raw generation followed by a separate transform.
  An independent same-device comparison improved the 1M kernel and full API
  time ratio by about 1.20x; the 262K full API had no consistent gain and the
  16M fused candidate remained slower than the two-stage path. The isolated
  production integration passed both seed cases with 130 comparisons each;
  its complete 520-point, eight-API run reached the 0.95 geometric-mean target
  for five APIs. Uniform FP64, log-normal FP32 and Poisson remained below
  target. This separate run does not replace parts of the full 32-API result.
  The later r5 integration adds bounded 32-call prefetch, large cache-only
  shared refills and the alias-table size limit described above. Four targeted
  tests passed 372 comparisons, and the SIZE2048 global fallback passed
  another 18 comparisons. The complete r5 snapshot then measured 1,888
  points across 32 APIs: 31 reached a 0.95 geometric mean, while MTGP32
  uniform FP64 remained at 0.702032. Four-engine Poisson was 1.410148 and
  the equal-API MTGP32 mean was 1.033515; family means do not replace
  the per-API target. The later r6 cache and runtime-copy implementation
  completed a separate MTGP32 eight-API run: its equal-API geometric mean
  was 1.154102, and its all-520-point mean was 1.105374. All eight APIs
  reached the target, but this source version has no complete 32-API run yet.
  Relative speedup changes between r5 and r6 include native-library timing
  changes. For the 35 uniform FP64 points at N <= 65536, the geometric mean
  of FlagRAND API time changed from 15.274 to 9.327 microseconds, while
  mcRAND changed from 10.811 to 16.855 microseconds. These temporal results
  are separate from the controlled same-process implementation comparisons.
  MetaX FP64 log-normal uses the measured degree13 exp polynomial in its
  separate, buffered and small aligned conversion paths. Range reduction,
  binary64 scaling and subnormal/NaN/Inf handling are unchanged. The default
  exp helper, XORWOW and other backends retain degree16. A 26,960-input C550
  Decimal80 comparison observed at most 1 ULP for both versions; this is a
  sampled precision result, not an all-input correct-rounding claim.

For stateful engines, the device state and cursor/cache determine continuation.
The displayed consumed-word count alone is not a serialization of an arbitrary
mixed API call history. Python cursor/cache updates are not captured by device
graph replay; these new APIs are measured using ordinary host calls.

## Poisson conversion

A cached Walker alias table is built from the Poisson PMF recurrence around
its mode. Each omitted tail has a geometric bound below `1e-18` times the
unnormalized mode weight. Integer alias thresholds introduce finite-resolution
rounding (about 32 bits); this is not a normal approximation. Lambda zero is
supported as a degenerate distribution. Tables and their device copies are
reused for fixed lambda.

Philox and XORWOW fuse raw generation and alias sampling. XORWOW fusion
preserves the preceding raw-then-transform output and final state exactly.
MRG32K3A uses fusion for up to 262144 outputs; larger calls use
separate generation and conversion because that path was faster on C550.
The two paths preserve the same output and continuation state. On MetaX,
MTGP32 short calls with `N <= 65536` use the bounded derived-cache path
above. Outside that path, calls that start and end on complete 256-word
updates fuse generation and conversion. Adjacent-word pairs share recurrence
loads before applying the alias conversion. Other calls use a separate
conversion kernel. That conversion reads the bounded raw cache directly;
refills retain the same generator sequence and remain inside ordinary
host-call timing, as do conversion rebuilds and device copies.
Two raw words
are consumed per Poisson output; MRG32K3A's smaller residue range is accounted
for in both bucket selection and thresholds.

## C550 reproduction and result format

Use the measured MACA image and installation commands in
[C550 environment and reproduction](c550-environment.md). The formal entry
is `tools/run_tests.py`; it runs accuracy and performance phases and writes
the unified `summary.json` plus the per-API `api_summary.json`.

From the prepared C550 container, run all 32 generator/API combinations:

```bash
export FLAGRAND_PLATFORM=metax
export FLAGRAND_PERF_PROFILE=acceptance
export FLAGRAND_OPS='raw32,uniform_f32,uniform_f64,normal_f32,normal_f64,lognormal_f32,lognormal_f64,xorwow,mrg32k3a,mtgp32,poisson'
"$PYTHON_BIN" tools/run_tests.py \
  --ops "$FLAGRAND_OPS" \
  --gpus 0 \
  --dump-output \
  --output-dir results/c550-full-32
```

Set `FLAGRAND_PERF_PROFILE=smoke` for three shapes per new API. The existing
Philox seven-API acceptance shapes stay unchanged. The new families use the
fixed matrix in `benchmark/rng_delivery/config.py`: 44 raw/uniform/Poisson
shapes, 42 normal/lognormal shapes, and five Poisson lambdas (0.1, 1, 10, 30,
1000). These new operator entries are currently at the `alpha` stage; the
explicit `--ops` list selects them without changing the default stable set.

The per-API measurement module is `benchmark.run_rng_delivery`. For a focused
reproduction of one API over its complete shape set:

```bash
"$PYTHON_BIN" -m benchmark.run_rng_delivery \
  --profile matrix \
  --distributions mtgp32_normal_f64 \
  --results-dir results/c550-mtgp32-normal-f64
```

This focused command saves raw records and a per-run summary. It does not
establish a complete eight-API family result; use the unified entry for the
complete delivery. Each raw run preserves `manifest.json`, `records.jsonl`,
`records.csv` and the individual timing samples. The manifest records the
configured shapes, lambda values, software environment and source hashes.

The primary metric is `total_ms_median`: repeated host API calls using a
persistent generator and preallocated output, with final synchronization.
The matrix profile uses five warmups, eleven timed groups and a 200 ms target
per group. Generator initialization, first compilation and first Poisson
table construction are outside this steady-state interval. Device event and
enqueue measurements remain separate fields.

Every point compares the same device, generator, distribution and output
precision against native mcRAND. Its speedup is `mcRAND_time / flagrand_time`.
The established `speedup` field remains an arithmetic mean. The new
`geometric_mean` and `api_geometric_means` fields explicitly report geometric
means over the configured points; Poisson retains every shape/lambda pair.
An eight-API family average gives each API equal weight, so Poisson contributes
once. `complete_api_family` records coverage, and `api_meets_target` records
whether each API reaches 0.95. A completed process or a family average above
0.95 does not imply every API has reached the target.

`tests/accuracy/test_rng_families.py` compares independent scalar recurrences,
state continuation, seed reset, current streams, tails, distribution moments
and sampled CDF values. Its `accuracy_result.json` observations preserve the
numerical measurements and criteria. These checks do not constitute a
statistical battery certification such as TestU01/BigCrush.

## Earlier device measurements

Earlier development measurements covered MetaX C550 (Torch
`2.8.0+metax3.7.0.7`, Triton `3.1.0`, mcRAND `100000`), Hygon BW1000
(DTK 25.04, Torch `2.4.1`, Triton `3.6.0`, hipRAND `300001`), PPU-ZW810E
(Torch `2.9.0`, Triton `3.6.0`, ACRAND `1400`) and MTT S5000
(MUSA 4.3.5, Torch `2.5.0`, torch_musa `2.5.0+f46274a`, Triton `3.1.0`,
muRAND `100105`). Results remain separate by device, generator and API.
Large speedups over a particular native library version do not establish a
ranking of the chips.

On S5000, import `torch_musa` before FlagRAND and allocate tensors on `musa`.
The implementation does not redirect `torch.cuda` calls. The earlier S5000
matrix contains 336 paired measurements and 48 repeatability/distribution
checks, without an API error; independent CPU raw/state comparisons passed
for XORWOW, MRG32K3A and MTGP32.

PPU's MTGP32 two-warp layout failed in `ppu-llc` on the measured stack. Small
calls use four warps. The earlier candidate matrix contains 84 timed rows
and 12 successful checks; raw/state comparisons also passed. All eight
corresponding ACRAND MTGP32 APIs aborted with `Unimplemented API`, so that
family had no native speedup. The other three PPU families have 252 completed
paired rows and 36 checks; their combined process later aborted at the first
native MTGP32 call.

Those development matrices used seven shapes from 1024 through 4194304,
with nine single-call device-event samples after four warmups. Event intervals
can include host submission gaps. They are a different measurement from the
complete host-API protocol above and do not establish its acceptance result.

The earlier C550 Poisson run on 2026-10-20 contains 70 paired rows and ten
seed repeatability/distribution checks. Across five lambda groups, seven-shape
geometric means were 1.50–1.58 for Philox and 1.66–1.73 for XORWOW. The earlier
C550 MRG32K3A r14 matrix contains 84 paired rows and 12 successful checks:
raw 1.850, uniform f32/f64 1.745/1.788, normal f32/f64 1.203/1.119,
lognormal f32/f64 1.154/1.021; Poisson lambda groups were 1.351–1.438.
The earlier C550 MTGP32 r19 family geometric mean was 0.877, below target.
These are historical measurements. Results from the complete C550 delivery
flow must be generated and reported with their own source version and shape
configuration.

## Sources

* [NVIDIA host API overview](https://docs.nvidia.com/cuda/curand/host-api-overview.html)
* [NVIDIA device API overview](https://docs.nvidia.com/cuda/curand/device-api-overview.html)
* [L'Ecuyer, Good Parameters and Implementations for Combined Multiple Recursive Random Number Generators](https://doi.org/10.1287/opre.47.1.159)
* [Saito and Matsumoto, Variants of Mersenne Twister Suitable for Graphic Processors](https://arxiv.org/abs/1005.4973)
* [rocRAND source](https://github.com/ROCm/rocm-libraries/tree/e08b5e451ed6fea566826b34eb7ec9eb18adb641/projects/rocrand)
* [FlagGems source and contribution structure](https://github.com/flagos-ai/FlagGems/tree/453da2f32f8a3798dea6cd3678922d39020f3e52)
