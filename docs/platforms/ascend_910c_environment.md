# Ascend 910C 从零构建与一键验收

## 交付内容

本目录从公开网络构建一个 `flagrand-ascend910c:cann9.0-flagtree3.5` 镜像。验收人员不需要公司内部镜像、内部软件源、预下载 wheel 或已有 Python 环境。

镜像固定包含以下软件：

| 组件 | 版本或来源 |
|---|---|
| 基础系统 | Ubuntu 22.04，公开 Docker Hub 镜像 |
| Python | 3.11.15，Python 官方源码 |
| CANN | 9.0.0 Toolkit 与 A3 ops，华为公开下载源 |
| PyTorch | 2.9.0+cpu，PyTorch 官方 aarch64 wheel |
| torch_npu | 2.9.0.post2，昇腾社区公开 aarch64 wheel |
| Triton | FlagTree `0.6.0+triton3.5` release 的 Ascend backend |
| FlagRAND | 当前交付包源码 |

最终只产生一个 FlagRAND 运行镜像。FlagRAND 不依赖 ATB、NNAL、vLLM、FlagGems、CUDA、cuDNN 或内部基础镜像。

## 节点要求

在 Ascend 910C aarch64 节点上准备以下宿主机环境：

1. 已安装与 CANN 9.0 兼容的 NPU driver 和 firmware。
2. 已安装 Docker 与 Ascend Docker Runtime，Docker runtime 名称为 `ascend`。
3. 构建阶段可以访问 Docker Hub、华为昇腾下载站、PyTorch、GitCode、GitHub 和 Python.org。
4. Docker 数据目录至少保留 40 GB 可用空间。

进入节点后先执行：

```bash
uname -m
npu-smi info
docker info | sed -n '/Runtimes/p'
```

预期架构为 `aarch64`，`npu-smi info` 能看到 910C，Docker runtimes 中包含 `ascend`。

## 第一步：解压源码

将交付压缩包上传到节点的 `/tmp`，然后执行：

```bash
mkdir -p /home/flagrand_ascend910c_delivery
tar -xzf /tmp/FlagRAND_Ascend910C_public_build_20260817.tar.gz \
    -C /home/flagrand_ascend910c_delivery
cd /home/flagrand_ascend910c_delivery/FlagRAND-pr1
```

## 第二步：构建镜像

```bash
bash docker/ascend910c/build.sh
```

该命令直接从公开源下载并安装全部依赖。构建结束时应显示：

```text
Ascend 910C image: flagrand-ascend910c:cann9.0-flagtree3.5
IMAGE BUILD: PASS
```

## 第三步：执行正式验收

```bash
bash docker/ascend910c/run.sh
```

该命令使用 0 号 NPU，依次执行环境记录、preflight、pytest 正确性检查以及七个 API 的独立进程端到端性能验收。正式阈值为每个 API 的聚合 speedup 不低于 `0.95x`。

成功结束时应显示：

```text
ASCEND 910C V6 STATUS: PASS
```

结果压缩包保存在：

```text
ascend_910c_delivery_results/<运行时间>/FlagRAND_910C_results_<运行时间>.tar.gz
```

使用其他 NPU 时执行：

```bash
FLAGRAND_DEVICE=1 bash docker/ascend910c/run.sh
```

## 文件职责

| 文件 | 用途 |
|---|---|
| `docker/ascend910c/Dockerfile` | 从 Ubuntu 22.04 构建完整 910C 运行环境 |
| `docker/ascend910c/build.sh` | 一键构建镜像 |
| `docker/ascend910c/run.sh` | 一键启动容器并持久化验收结果 |
| `docker/ascend910c/entrypoint.sh` | 容器启动时加载 CANN 9.0 环境 |
| `scripts/delivery/run_ascend_910c.sh` | 容器内正式验收入口 |
| `scripts/delivery/run_ascend_910c_v6.sh` | 已完成硬件验证的 v6 回归入口 |
