# FlagRAND Philox 沐曦运行指南

PPU-ZW810E 请看 [PPU 运行指南](ppu.md)，使用 `benchmark.run_ppu` 和 `benchmark/ppu_matrix/`。
本文件保留给沐曦/MACA 路径，入口是 `benchmark.run_muxi`。

本次测试对象固定为：

| Host API | 输出 dtype | 数量约束 |
|---|---|---:|
| `generate` | `int32` | 任意非负整数 |
| `generate_uniform` | `float32` | 任意非负整数 |
| `generate_uniform_double` | `float64` | 任意非负整数 |
| `generate_normal` | `float32` | 偶数 |
| `generate_normal_double` | `float64` | 2 的倍数 |
| `generate_lognormal` | `float32` | 偶数 |
| `generate_lognormal_double` | `float64` | 2 的倍数 |

Muxi 入口默认使用沐曦原生 `libmcrand.so` 作为七类 Host API 的性能 baseline。
Torch 数字仍只作为同一设备上的平台参考，不表示算法等价。PPU cuRAND baseline
默认跳过；如需强制运行，可加 `--with-ppu-curand`。

## 1. 准备代码

git clone 后进入 `FlagRAND` 项目根目录。

项目根目录应至少包含：

```text
FlagRAND/
├── src/flagrand/
├── tests/
├── benchmark/muxi_matrix/
├── benchmark/run_muxi.py
├── docs/usage/non-nvidia/muxi.md
└── pyproject.toml
```

不要传输本机 conda 环境、Triton cache、`muxi_results/` 或构建产物。

## 2. 启动沐曦容器

宿主目录可按实际位置修改：

```bash
docker run --rm -it \
  --device=/dev/dri \
  --device=/dev/mxcd \
  -v "$HOME/xuzc:/workspace" \
  -w /workspace/FlagRAND \
  cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.0-maca.ai3.7.0.107-torch2.8-py310-ubuntu22.04-amd64 \
  bash
```

在容器内固定从当前目录导入，并把缓存也留在项目目录中：

```bash
export PYTHONPATH="$PWD/src:$PWD"
export TRITON_CACHE_DIR="$PWD/.triton_cache"
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
```

不建议执行 `pip install triton` 或安装 NVIDIA 版 PyTorch。镜像自带的 MACA
PyTorch/Triton 才是这里需要验证的运行时。

## 3. 确认导入来源

先确认测试没有误导入服务器上的其他 FlagRand：

```bash
python - <<'PY'
import flagrand
import torch
from triton.runtime import driver

print("flagrand:", flagrand.__file__)
print("torch:", torch.__version__)
print("device:", torch.cuda.get_device_name(0))
print("target:", driver.active.get_current_target())
PY
```

`flagrand` 路径必须落在当前目录的 `src/flagrand` 下，target 的 backend 应为
`maca`。脚本还会再次检查导入路径并记录全部 Python 源码的 SHA-256。

确认当前运行环境能加载 mcRAND：

```bash
ldconfig -p | grep libmcrand.so
python3 - <<'PY'
from benchmark.muxi_matrix.mcrand import mcrand_library_path, mcrand_version

print("mcRAND library:", mcrand_library_path())
print("mcRAND version:", mcrand_version())
PY
```

默认搜索 `$MACA_PATH/lib/libmcrand.so`、`/opt/maca/lib/libmcrand.so` 和
`/opt/maca-*/lib/libmcrand.so`。非标准路径可通过 `FLAGRAND_MCRAND_PATH` 指定。

## 4. 正确性检验

```bash
python3 -m benchmark.run_muxi --preflight-only
```

检验会依次完成：

1. 编译并执行 seed=0 的 Philox raw32 已知向量；
2. 对七种输出分别验证非 4 倍数 split，确认连续调用后的任意 offset 正确；
3. 验证 seed、offset 和 Host API 状态推进在沐曦后端上没有被破坏。

预期输出包含：

```text
Preflight: pass
Results: .../muxi_results/<timestamp>_smoke_philox
```

如果预检失败，不要继续看性能。先保留结果目录中的 `manifest.json`，它包含导入来源、
运行时版本、Triton target、已启用的兼容补丁和具体失败项。



## 5. mcRAND smoke baseline

先运行三组 shape 的七类 API，关闭无关的 Torch 参考：

```bash
python3 -m benchmark.run_muxi \
  --profile smoke \
  --skip-torch-reference
```

每个 shape 会依次运行 FlagRAND 和 mcRAND，并报告：

```text
speedup_vs_mcrand = mcRAND total_ms_median / FlagRAND total_ms_median
```

结果同时记录 mcRAND 动态库路径、版本、Host 总耗时、设备事件耗时和 enqueue 耗时。
如需只验证 FlagRAND，可加 `--skip-mcrand`。

## 6. 完整矩阵

```bash
python3 -m benchmark.run_muxi --profile full
```

full 的元素数量为：

```text
4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
```

产生 49 个 FlagRAND case。启用 Torch 参考时，为每种规模运行六个 RNG 参考和四个
纯写参考。需要临时缩小范围时可以使用：

```bash
python3 -m benchmark.run_muxi \
  --profile smoke \
  --distributions raw32,uniform_f32,normal_f32 \
  --sizes 4096,65536 \
  --skip-torch-reference
```

`--sizes` 中的数字只要求为正数；normal/lognormal 的奇数长度会被跳过。泛用性组可
分两次运行：

```bash
python3 -m benchmark.run_muxi \
  --profile smoke \
  --distributions raw32,uniform_f32,uniform_f64 \
  --sizes 1,2,3,5,7,4097,65537,1048577 \
  --skip-torch-reference

python3 -m benchmark.run_muxi \
  --profile smoke \
  --distributions normal_f32,normal_f64,lognormal_f32,lognormal_f64 \
  --sizes 2,6,4098,65538,1048578 \
  --skip-torch-reference
```

如果需要在 Muxi 入口里临时跑 PPU cuRAND，可加 `--with-ppu-curand`，通常不建议。

## 7. LibTuner A/B

手工配置基线：

```bash
export FLAGRAND_USE_LIBTUNER=0
python3 -m benchmark.run_muxi \
  --profile full \
  --skip-torch-reference \
  --results-dir muxi_results/manual
```

第一次 LibTuner 运行用于搜索，不能直接与基线比较：

```bash
export FLAGRAND_USE_LIBTUNER=1
export TRITON_PRINT_AUTOTUNING=1
python3 -m benchmark.run_muxi \
  --profile full \
  --skip-torch-reference \
  --results-dir muxi_results/libtuner_warmup
```

第二次读取缓存后正式计时：

```bash
export FLAGRAND_USE_LIBTUNER=1
unset TRITON_PRINT_AUTOTUNING
python3 -m benchmark.run_muxi \
  --profile full \
  --skip-torch-reference \
  --results-dir muxi_results/libtuner_hot
```

## 8. 沐曦兼容层做了什么

兼容逻辑在 Triton target backend 为 `maca` 时有效：

1. 强制使用标准 Triton JIT launch，绕开依赖 NVIDIA Triton 私有 launcher ABI 的缓存启动器；
2. 将 f32 Box-Muller 的 NVIDIA inline PTX `sin/cos` 替换成 `tl.sin/tl.cos`；
3. 保留 Philox counter、raw 到 uniform 的映射、f64 多项式路径和 Host API 状态语义。

因此验收目标是
Philox 流推进一致、分段与整段一致，并通过分布统计检查。

## 9. 结果目录

每次运行都会新建：

```text
muxi_results/<timestamp>_<profile>_philox/
├── manifest.json
├── records.jsonl
├── records.csv
└── summary.md
```

- `manifest.json`：环境、源码哈希、兼容补丁和 preflight；
- `records.jsonl`：保留嵌套的检查项与错误栈；
- `records.csv`：方便筛选和画图；
- `summary.md`：适合直接审阅的汇总表。

把结果带回本机时只需压缩该次目录：

```bash
RESULT_DIR="$(find muxi_results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tar -czf muxi_philox_results.tar.gz "$RESULT_DIR"
```
