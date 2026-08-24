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
| Triton | FlagTree `5858e827ed0c66cebd5248e0200edb81cdd838a4`（`0.6.0+triton3.5` release）的 Ascend backend |
| FlagRAND | 当前交付包源码 |

最终只产生一个 FlagRAND 运行镜像。FlagRAND 不依赖 ATB、NNAL、vLLM、FlagGems、CUDA、cuDNN 或内部基础镜像。

## 节点要求

在 Ascend 910C aarch64 节点上准备以下宿主机环境：

1. 已安装与 CANN 9.0 兼容的 NPU driver 和 firmware。
2. 已安装 Docker 与 Ascend Docker Runtime，Docker runtime 名称为 `ascend`。
3. 构建阶段可以访问 Docker Hub、华为昇腾下载站、PyTorch、GitCode、GitHub 和 Python.org。
4. Docker 数据目录至少保留 40 GB 可用空间。

## 直接验收

克隆交付分支并执行唯一入口：

```bash
git clone \
    --branch main \
    --single-branch \
    git@github.com:flagos-ai/FlagRAND.git \
    /home/flagrand_910c_acceptance

cd /home/flagrand_910c_acceptance
bash run_ascend_910c_acceptance.sh
```

脚本依次完成完整镜像构建和七 API 正式验收。构建与运行日志、结果压缩包统一保存在：

```text
ascend_910c_delivery_results/<运行时间>/
```

最终成功标志为：

```text
IMAGE BUILD: PASS
ASCEND 910C V6 STATUS: PASS
FLAGRAND ASCEND 910C DELIVERY: PASS
```

## 使用网页上传的源码包

将交付压缩包上传到节点的 `/tmp`，然后执行：

```bash
mkdir -p /home/flagrand_ascend910c_delivery
tar -xzf /tmp/FlagRAND_Ascend910C_public_build_20260817.tar.gz \
    -C /home/flagrand_ascend910c_delivery \
    --strip-components=1
cd /home/flagrand_ascend910c_delivery
bash run_ascend_910c_acceptance.sh
```

使用其他 NPU 时执行：

```bash
FLAGRAND_DEVICE=1 bash run_ascend_910c_acceptance.sh
```

## 文件职责

| 文件 | 用途 |
|---|---|
| `run_ascend_910c_acceptance.sh` | 从构建镜像到正式验收的唯一宿主机入口 |
| `docker/ascend910c/Dockerfile` | 从 Ubuntu 22.04 构建完整 910C 运行环境 |
| `docker/ascend910c/build.sh` | 一键构建镜像 |
| `docker/ascend910c/run.sh` | 一键启动容器并持久化验收结果 |
| `docker/ascend910c/entrypoint.sh` | 容器启动时加载 CANN 9.0 环境 |
| `scripts/delivery/run_ascend_910c.sh` | 容器内正式验收入口 |
| `scripts/delivery/run_ascend_910c_v6.sh` | 已完成硬件验证的 v6 回归入口 |
