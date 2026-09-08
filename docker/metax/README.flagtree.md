# MetaX C550：Python 3.12 / FlagTree Triton 3.6 环境

该环境用于验证 MetaX launcher 在无旧版 `JITFunction.binder` 接口的 Triton 上运行。
它以节点现有的 Python 3.12 / MACA 3.7.2 基础镜像为底，安装
`flagtree===0.6.1+metax3.6`，由该包提供 `triton` 模块。基础镜像中的 MetaX Torch 保持原样。

基础镜像和 FlagTree 安装版本来自
[FlagTree 官方 MetaX 手册](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-metax)。
安装包索引已确认存在 `flagtree-0.6.1+metax3.6-cp312-cp312-linux_x86_64.whl`。
这是可复现的新环境配置；原验收截图没有给出 FlagTree/FlagGems 的准确版本，不能视为原环境的逐包复制。

## 构建

在宿主机 FlagRAND 仓库根目录执行：

```bash
docker build \
  --network=host \
  --platform=linux/amd64 \
  -f docker/metax/Dockerfile.flagtree \
  -t flagrand-metax:py312-flagtree0.6.1-metax3.6 \
  .
```

Dockerfile 中安装依赖和导入检查均显式使用 `/opt/conda/bin/python`，无需在交互 shell 中激活
Conda。构建阶段打印 Python、Torch、FlagTree 和 Triton 的版本，不进行 GPU 初始化。

## 启动与验证

从宿主机项目根目录创建容器：

```bash
docker run -it \
  --name flagrand-muxi-py312 \
  --device=/dev/mxcd \
  --device=/dev/dri \
  -v "$PWD:/workspace/FlagRAND" \
  -w /workspace/FlagRAND \
  flagrand-metax:py312-flagtree0.6.1-metax3.6
```

容器中运行：

```bash
/opt/conda/bin/python - <<'PY'
import sys
import importlib.metadata
import torch
import triton
from triton.runtime import driver
from flagrand.metax.philox import _RAW_LAUNCHER

print("Python:", sys.executable, sys.version)
print("Torch:", torch.__version__)
print("FlagTree:", importlib.metadata.version("flagtree"))
print("Triton:", triton.__version__, triton.__file__)
print("Target:", driver.active.get_current_target())
print("Legacy binder:", hasattr(_RAW_LAUNCHER._kernel, "binder"))
PY

/opt/conda/bin/python -m pytest -q \
  tests/accuracy/test_philox_cuda.py -m lognormal_f64

/opt/conda/bin/python -m pytest -q \
  tests/accuracy/test_philox_cuda.py \
  tests/accuracy/test_philox_metax.py

/opt/conda/bin/python -m benchmark.run_muxi --preflight-only
```

预期版本为 Python 3.12、FlagTree `0.6.1+metax3.6`、Triton 3.6，target 为 `maca`。
`Legacy binder: False` 用于确认当前验证的是无旧版 binder 的启动路径。
新环境的硬件测试与性能结果需单独记录，不能沿用 Python 3.10 / MACA Triton 3.0 的实测数值。

后续进入同一容器：

```bash
docker start -ai flagrand-muxi-py312
```
