# FlagRand Philox 部分 PR：沐曦运行指南

本次测试对象固定为：

| Host API | 输出 dtype | 数量约束 |
|---|---|---:|
| `generate` | `int32` | 4 的倍数 |
| `generate_uniform` | `float32` | 4 的倍数 |
| `generate_uniform_double` | `float64` | 2 的倍数 |
| `generate_normal` | `float32` | 4 的倍数 |
| `generate_normal_double` | `float64` | 2 的倍数 |
| `generate_lognormal` | `float32` | 4 的倍数 |
| `generate_lognormal_double` | `float64` | 2 的倍数 |

Torch 数字只作为同一设备上的平台参考，不表示算法等价。

## 1. 准备代码

git clone 后进入 `FlagRAND` 项目根目录。

项目根目录应至少包含：

```text
FlagRAND/
├── src/flagrand/
├── tests/
├── muxi_matrix/
├── muxi_flagrand_matrix.py
├── MUXI_RUN_GUIDE.md
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

## 4. 正确性检验

```bash
python muxi_flagrand_matrix.py --preflight-only
```

检验会依次完成：

1. 编译并执行 seed=0 的 Philox raw32 已知向量；
2. 对七种输出分别验证一次生成 4096 个元素，是否与连续两次各生成 2048 个元素完全一致；
3. 验证 seed、offset 和 Host API 状态推进在沐曦后端上没有被破坏。

预期输出包含：

```text
Preflight: pass
Results: .../muxi_results/<timestamp>_smoke_philox
```

如果预检失败，不要继续看性能。先保留结果目录中的 `manifest.json`，它包含导入来源、
运行时版本、Triton target、已启用的兼容补丁和具体失败项。



## 5. 完整矩阵

```bash
python muxi_flagrand_matrix.py --profile full
```

full 的元素数量为：

```text
4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
```

产生 49 个 FlagRand case；启用 Torch 参考时，为每种规模运行六个 RNG
参考和四个纯写参考。需要临时缩小范围时可以使用：

```bash
python muxi_flagrand_matrix.py \
  --profile smoke \
  --distributions raw32,uniform_f32,normal_f32 \
  --sizes 4096,65536 \
  --skip-torch-reference
```

`--sizes` 中的数字必须为正数且能被 4 整除，这样七种 API 都满足当前版本的
向量化约束。

## 6. 沐曦兼容层做了什么

兼容逻辑在 Triton target backend 为 `maca` 时有效：

1. 强制使用标准 Triton JIT launch，绕开依赖 NVIDIA Triton 私有 launcher ABI 的缓存启动器；
2. 将 f32 Box-Muller 的 NVIDIA inline PTX `sin/cos` 替换成 `tl.sin/tl.cos`；
3. 保留 Philox counter、raw 到 uniform 的映射、f64 多项式路径和 Host API 状态语义。

因此验收目标是
Philox 流推进一致、分段与整段一致，并通过分布统计检查。

## 7. 结果目录

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
tar -czf muxi_philox_results.tar.gz muxi_results/<本次目录名>
```
