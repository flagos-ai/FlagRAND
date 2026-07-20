# FlagRAND Philox PPU-ZW810E Run Guide

PPU entry point:

```bash
python ppu_flagrand_matrix.py
```

PPU-specific baseline wrappers live under:

```text
ppu_matrix/
```

The Muxi benchmark harness remains under:

```text
muxi_matrix/
```

## 1. Environment

```bash
export PYTHONPATH="$PWD/src:$PWD"
unset FLAGRAND_FORCE_PORTABLE
unset FLAGRAND_USE_LIBTUNER

export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p ppu_results ppu_logs
```

Formal performance runs should not set `FLAGRAND_FORCE_PORTABLE=1`; that flag disables the cached binary launcher and is intended only for compatibility debugging.

## 2. Correctness

```bash
python ppu_flagrand_matrix.py --preflight-only \
  2>&1 | tee ppu_logs/preflight.log

python -m pytest -q tests/test_api.py tests/test_philox_cuda.py \
  2>&1 | tee ppu_logs/pytest.log
```

Expected result: preflight passes and pytest reports all tests passed.

## 3. Performance

Raw/Uniform odd and boundary shapes:

```bash
python ppu_flagrand_matrix.py \
  --profile smoke \
  --distributions raw32,uniform_f32,uniform_f64 \
  --sizes 1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096,4097,65535,65536,65537,1048575,1048576,1048577 \
  --skip-torch-reference \
  --results-dir ppu_results/generic_uniform_wide \
  2>&1 | tee ppu_logs/generic_uniform_wide.log
```

Normal/LogNormal even, non-4-multiple, and boundary shapes:

```bash
python ppu_flagrand_matrix.py \
  --profile smoke \
  --distributions normal_f32,normal_f64,lognormal_f32,lognormal_f64 \
  --sizes 2,4,6,8,10,14,16,18,126,128,130,1022,1024,1026,4094,4096,4098,65534,65536,65538,1048574,1048576,1048578 \
  --skip-torch-reference \
  --results-dir ppu_results/generic_normal_wide \
  2>&1 | tee ppu_logs/generic_normal_wide.log
```

`ppu_flagrand_matrix.py` runs both PPU `libcurand.so` compatibility and native `libacrand.so` baselines by default. FlagRAND records include `speedup_vs_ppu_curand` and `speedup_vs_ppu_acrand`.

## 4. LibTuner A/B

Manual baseline:

```bash
unset FLAGRAND_USE_LIBTUNER
python ppu_flagrand_matrix.py \
  --profile full \
  --skip-torch-reference \
  --results-dir ppu_results/manual
```

LibTuner warmup/search:

```bash
export FLAGRAND_USE_LIBTUNER=1
export TRITON_PRINT_AUTOTUNING=1
python ppu_flagrand_matrix.py \
  --profile full \
  --skip-torch-reference \
  --results-dir ppu_results/libtuner_warmup
```

LibTuner cached timing:

```bash
export FLAGRAND_USE_LIBTUNER=1
unset TRITON_PRINT_AUTOTUNING
python ppu_flagrand_matrix.py \
  --profile full \
  --skip-torch-reference \
  --results-dir ppu_results/libtuner_hot
```
