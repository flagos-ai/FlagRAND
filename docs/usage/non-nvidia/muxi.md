# FlagRAND 沐曦 MetaX C550 / MACA 运行指南

PPU-ZW810E 请看 [PPU 运行指南](ppu.md)，使用 `benchmark.run_ppu` 和 `benchmark/ppu_matrix/`。
本文件保留给沐曦 MetaX/MACA 路径，canonical vendor 是 `metax`，Triton target
backend 是 `maca`，底层矩阵入口是 `benchmark.run_muxi`。从新 `main` 克隆后的容器启动、
统一 runner 验收和 `summary.json` 校验见 [硬件验收流程](../../getting-started/hardware-acceptance.md)。

> 验证状态：本项目已在 pre-main 沐曦分支上完成 C550 的 216 条矩阵验收。NVIDIA 新
> main 和 MetaX 独立分流的集成版尚未在 C550 重测；旧性能数值不代表集成后 `main`
> 已通过。

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

宿主机从远端 `main` 新克隆后进入 `FlagRAND` 项目根目录，并在宿主机记录版本：

```bash
git clone --branch main --single-branch \
  git@github.com:flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

项目根目录应至少包含：

```text
FlagRAND/
├── docker/metax/
│   ├── Dockerfile
│   ├── build.sh
│   ├── entrypoint.sh
│   └── run.sh
├── src/flagrand/
├── tests/
├── benchmark/muxi_matrix/
├── benchmark/run_muxi.py
├── docs/usage/non-nvidia/muxi.md
└── pyproject.toml
```

不要传输本机 conda 环境、Triton cache、`muxi_results/` 或构建产物。

## 2. 构建并启动沐曦容器

已验证的基础镜像包含 MACA PyTorch/Triton 和原生 mcRAND：

```text
cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.0-maca.ai3.7.0.107-torch2.8-py310-ubuntu22.04-amd64
```

基础镜像不包含 Git。仓库提供的 `docker/metax/Dockerfile` 以该镜像为基础，保留其中的
Torch、Triton、MACA runtime 和 mcRAND，只增加 Git/SSH、PyYAML、pytest、FlagRAND 源码及
固定的验收环境变量。在项目根目录构建开发镜像：

```bash
export FLAGRAND_METAX_IMAGE='flagrand-metax:c550-maca3.7.0.107'
bash docker/metax/build.sh
```

如需使用另一个 MACA 基础镜像：

```bash
export FLAGRAND_METAX_BASE_IMAGE='<MetaX MACA base image>'
bash docker/metax/build.sh
```

首次创建持久开发容器：

```bash
bash docker/metax/run.sh
```

`run.sh` 传入 `/dev/mxcd` 和 `/dev/dri`，并将当前项目目录挂载到
`/workspace/FlagRAND`。默认容器名是 `flagrand-muxi-dev`，后续进入同一容器：

```bash
docker start -ai flagrand-muxi-dev
```

自定义容器名时，构建和启动示例为：

```bash
export FLAGRAND_METAX_CONTAINER='flagrand-muxi-dev-wuyu'
bash docker/metax/run.sh
```

镜像默认把当前目录源码加入 `PYTHONPATH`，设置 `FLAGRAND_VENDOR=metax`，并把 Triton
缓存留在项目目录中。入口脚本还会加载 `scripts/ci/backend_env.sh metax`。进入容器后
这些值等价于：

```bash
export PYTHONPATH="$PWD/src:$PWD"
export TRITON_CACHE_DIR="$PWD/.triton_cache"
export XDG_CACHE_HOME="$PWD/.cache"
export PYTHON_BIN=python3
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
```

上述 MACA 3.7.0 镜像使用其预装的 PyTorch/Triton。
Python 3.12 / Triton 3.6 的新环境使用
[FlagTree MetaX 容器配置](../../../docker/metax/README.flagtree.md)。该 py312 基础镜像的
`/opt/conda/bin/python` 已提供 Torch，但需要安装官方 MetaX 版 FlagTree 来提供 Triton；
新 Dockerfile 已包含这一步。

## 3. 确认导入来源

先确认测试没有误导入服务器上的其他 FlagRand：

```bash
python3 - <<'PY'
import flagrand
import torch
from triton.runtime import driver

print("flagrand:", flagrand.__file__)
print("torch:", torch.__version__)
print("available:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0))
print("target:", driver.active.get_current_target())
PY
```

`flagrand` 路径必须落在当前目录的 `src/flagrand` 下，target 的 backend 应为
`maca`。已验证的镜像显示 `torch==2.8.0+metax3.7.0.7`、`MetaX C550` 和
`GPUTarget(backend='maca', arch=80, warp_size=64)`。脚本还会再次检查导入路径并记录
全部 Python 源码的 SHA-256。

确认当前运行环境能加载 mcRAND：

```bash
ldconfig -p | grep -E 'libmcrand|libmcruntime'
python3 - <<'PY'
from benchmark.muxi_matrix.mcrand import mcrand_library_path, mcrand_version

print("mcRAND library:", mcrand_library_path())
print("mcRAND version:", mcrand_version())
PY
```

默认搜索 `$MACA_PATH/lib/libmcrand.so`、`/opt/maca/lib/libmcrand.so` 和
`/opt/maca-*/lib/libmcrand.so`。已验证镜像返回 `/opt/maca/lib/libmcrand.so` 和版本值
`100000`。非标准路径可通过 `FLAGRAND_MCRAND_PATH` 指定。

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

## 6. 正式 acceptance 和完整矩阵

统一 runner 的 canonical backend 名是 `metax`：

```bash
export FLAGRAND_RUN_ID="metax_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN=python3 \
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh metax
export SUMMARY="$PWD/muxi_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

它会执行环境记录、统一 runner 自测、正确性和以 mcRAND 为 baseline 的性能矩阵，并在
验收根目录生成 FlagOS `summary.json`。

直接运行同一进程的 MetaX 原始 acceptance 矩阵：

```bash
python3 -m benchmark.run_muxi \
  --profile full \
  --acceptance-profile \
  --acceptance-baseline mcrand \
  --with-mcrand \
  --acceptance-min 0.95 \
  --acceptance-max 1.50 \
  --strict-acceptance \
  --skip-torch-reference
```

固定 acceptance shape 集包含 108 个 FlagRAND case 和 108 个 mcRAND case，共 216 条记录。
性能判定先计算每个 shape 的 `mcRAND total_ms_median / FlagRAND total_ms_median`，再在单个
API 内取算术平均，不计算跨 API 平均值。通过时输出包含：

```text
Acceptance status: pass (7/7 APIs in target range)
Completed 216 records with 0 failure(s).
```

pre-main 沐曦分支已通过该矩阵；新 main 集成版必须在 C550 上重新生成这两行结果，
之前的加速比不作为新 main 的验收结论。

不带 `--acceptance-profile` 的普通 full 矩阵用于诊断固定宽 shape：

```bash
python3 -m benchmark.run_muxi \
  --profile full \
  --skip-torch-reference
```

full 的元素数量为：

```text
4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
```

默认产生 49 个 FlagRAND case 和 49 个 mcRAND case。启用 Torch 参考时，为每种规模
额外运行 RNG 和纯写参考。需要临时缩小范围时可以使用：

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

## 8. MetaX 独立分流和兼容层

当 vendor 为 `metax`、Triton target backend 为 `maca` 时：

1. 七个 Host API 先经过 `src/flagrand/runtime/backend/_metax/ops.py` 平台分流，不进入 NVIDIA 专属
   dispatcher；
2. raw、f32、f64 Philox kernel 全部位于 `src/flagrand/metax/philox.py`，不导入共享
   `rng/philox.py`、`philox_direct*` 或 NVIDIA kernel；
3. `src/flagrand/metax/transforms.py` 固定 C550 已验证的 uniform 与 Box–Muller 变换，f32
   normal/lognormal 使用 `tl.log`、`tl.sqrt`、`tl.sin`、`tl.cos` 可移植数学路径；
4. `src/flagrand/metax/launcher.py` 首次 JIT 编译后缓存 `CompiledKernel`，通过 MACA Triton
   旧版 binder specialization key
   复用编译结果，并直接调用 `CompiledKernel.run`；
5. 保留 Philox counter、Host API generator seed/offset 推进、bulk/split 和 replay 语义。

这个边界使 MetaX 的 kernel、数学语义和 MACA launcher 均可独立演进。验收目标是 Philox
流推进一致、分段与整段一致，并通过分布统计检查。

## 9. 结果目录

每次运行都会新建：

```text
muxi_results/<timestamp>_<profile>_philox/
├── manifest.json
├── records.jsonl
├── records.csv
├── summary.json
└── summary.md
```

- `manifest.json`：环境、源码哈希、兼容补丁和 preflight；
- `records.jsonl`：保留嵌套的检查项与错误栈；
- `records.csv`：方便筛选和画图；
- `summary.json`：与其他芯片一致的 FlagOS 机器可读汇总；
- `summary.md`：适合直接审阅的汇总表。

统一 runner 还会在 `muxi_results/ci_<RUN_ID>/` 生成跨 API 的 `summary.json`，并把上述
原始矩阵目录保留在每个 API 的 `performance_raw/` 下。

把结果带回本机时只需压缩该次目录：

```bash
RESULT_DIR="$(find muxi_results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tar -czf muxi_philox_results.tar.gz "$RESULT_DIR"
```

[性能正确性与实现说明](MetaX_C550_FlagRAND_性能正确性与实现说明.md) 中的实测数值来自 pre-main
沐曦分支；新 main 在 C550 重测前，该数值只作历史对照。
