# 阶段四：PPU 与 Ascend 910C 一键交付回归

本文件对应交付包 `FlagRAND_PPU_910C_stage4_v6_20260811.tar.gz`，顶层目录固定为
`FlagRAND-pr1/`。上传时先通过 JumpServer 放到目标节点 `/tmp/`，再解压到新的部署目录；命令会拒绝覆盖已有目录。

正式性能指标为端到端 `total_ms_median`。每个 case 先 warmup，再多次测量取中位数；每个 shape
的 speedup 为厂商原生 baseline 除以 FlagRAND，然后只在单 API 内对不同 shape 取算术平均，不计算跨
API 平均值。PPU 与 910C 均要求七个 API 各自达到 `0.95x`。

v6 保留已经通过实机验收的 kernel 数学实现和平台隔离，并在 v5 基础上固化 910C
Normal/LogNormal FP32 的 public-API host fast path：

- PPU kernel、PPU 3.6 编译兼容层及七 API acceptance 矩阵保持不变。
- 910C `normal_f32`、`lognormal_f32` 的常见路径在公共校验后直接进入 cached launcher；跨
  uint32 counter epoch 的极少数调用仍回退到完整 windowed 路径。
- 910C 只保留逐 API 独立进程的正式验收，删除同进程七 API 性能门禁。
- 一键脚本负责环境核验、静态编译、正确性预检、pytest、正式性能、汇总和结果打包。
- 两个平台使用独立的 v6 Triton cache，禁止复用旧交付包的编译产物。

## 1. PPU-ZW810E

### 1.1 从 `/tmp` 核验并解压

在 PPU 节点执行：

```bash
set -euo pipefail

archive=/tmp/FlagRAND_PPU_910C_stage4_v6_20260811.tar.gz
artifact_dir=/home/flagrand_delivery_artifacts
delivery_root=/home/flagrand_stage4_ppu_20260811_v6

test -f "$archive"
sha256sum "$archive"
tar -tzf "$archive" >/dev/null

if [[ -e "$delivery_root" ]]; then
    echo "Refusing to overwrite existing directory: $delivery_root" >&2
    exit 20
fi

install -d -m 0755 "$artifact_dir" "$delivery_root"
mv "$archive" "$artifact_dir/"
tar --no-same-owner -xzf "$artifact_dir/$(basename "$archive")" -C "$delivery_root"

project="$delivery_root/FlagRAND-pr1"
test -f "$project/pyproject.toml"
test -x "$project/scripts/delivery/run_ppu_v6.sh"

if find "$project" -name '._*' -o -name '.DS_Store' -o -name '__pycache__' | grep -q .; then
    echo "Unexpected metadata/cache file in delivery" >&2
    exit 21
fi

echo "PPU project: $project"
```

### 1.2 一键执行正式回归

```bash
set -euo pipefail
project=/home/flagrand_stage4_ppu_20260811_v6/FlagRAND-pr1
cd "$project"

FLAGRAND_PERF_PROFILE=acceptance \
    bash scripts/delivery/run_ppu_v6.sh
```

脚本必须以 `PPU V6 STATUS: PASS` 结束，同时输出结果压缩包及 SHA256。正式日志中还必须满足：

- `Preflight: pass`；
- pytest 无失败；
- `Acceptance status: pass`；
- 七个 `ACCEPTANCE` 汇总行全部为 `pass`。

结果压缩包位于 `/tmp/FlagRAND_PPU_v6_results_<时间戳>.tar.gz`，可直接由 JumpServer 下载。

## 2. Huawei Ascend 910C

910C 正式容器固定为 `FlagBLAS_ywb`，CANN 固定为 `/usr/local/Ascend/cann-9.0.0`。

### 2.1 从宿主 `/tmp` 复制进容器并解压

在 910C 节点宿主机执行：

```bash
set -euo pipefail

archive=/tmp/FlagRAND_PPU_910C_stage4_v6_20260811.tar.gz
container=FlagBLAS_ywb
container_archive=/home/FlagRAND_PPU_910C_stage4_v6_20260811.tar.gz
delivery_root=/home/flagrand_stage4_910c_20260811_v6

test -f "$archive"
sha256sum "$archive"
tar -tzf "$archive" >/dev/null
docker inspect "$container" >/dev/null
docker cp "$archive" "$container:$container_archive"

docker exec -i "$container" bash -s -- "$container_archive" "$delivery_root" <<'BASH'
set -euo pipefail
archive="$1"
delivery_root="$2"

if [[ -e "$delivery_root" ]]; then
    echo "Refusing to overwrite existing directory: $delivery_root" >&2
    exit 20
fi

install -d -m 0755 "$delivery_root"
tar --no-same-owner -xzf "$archive" -C "$delivery_root"
project="$delivery_root/FlagRAND-pr1"
test -f "$project/pyproject.toml"
test -x "$project/scripts/delivery/run_ascend_910c_v6.sh"

if find "$project" -name '._*' -o -name '.DS_Store' -o -name '__pycache__' | grep -q .; then
    echo "Unexpected metadata/cache file in delivery" >&2
    exit 21
fi

echo "910C project: $project"
BASH
```

### 2.2 一键执行逐 API 正式回归

在 910C 节点宿主机执行：

```bash
docker exec -i FlagBLAS_ywb bash <<'BASH'
set -euo pipefail
project=/home/flagrand_stage4_910c_20260811_v6/FlagRAND-pr1
cd "$project"

FLAGRAND_PERF_PROFILE=acceptance \
FLAGRAND_DEVICE=0 \
FLAGRAND_SPEEDUP_MIN=0.95 \
    bash scripts/delivery/run_ascend_910c_v6.sh
BASH
```

这是 910C 唯一的正式性能验收入口。脚本会按
`raw32 → uniform_f32 → uniform_f64 → normal_f32 → normal_f64 → lognormal_f32 → lognormal_f64`
依次启动七个独立 Python 进程。每个进程使用完整 acceptance shape、full warmup/repeat 和同一条
`0.95x` 单 API 门禁；不运行、不报告同进程七 API 性能结论。

脚本必须以 `ASCEND 910C V6 STATUS: PASS` 结束，同时满足：

- `Preflight: pass`；
- pytest 无失败；
- 七个独立日志各有且只有一个对应的 `API pass` 汇总行；
- `preflight_rc=0`、`pytest_rc=0`、`performance_rc=0`。

### 2.3 把 910C 结果复制到宿主 `/tmp`

一键脚本会在容器 `/home` 下生成 `FlagRAND_910C_v6_results_<时间戳>.tar.gz`。执行：

```bash
set -euo pipefail
container=FlagBLAS_ywb

container_archive="$(
    docker exec "$container" bash -lc \
        'ls -1t /home/FlagRAND_910C_v6_results_*.tar.gz | head -1'
)"
test -n "$container_archive"

host_archive="/tmp/$(basename "$container_archive")"
docker cp "$container:$container_archive" "$host_archive"
chmod 0644 "$host_archive"
tar -tzf "$host_archive" >/dev/null
ls -lh "$host_archive"
sha256sum "$host_archive"
echo "JumpServer download: $host_archive"
```

## 3. 快速 smoke（非交付门禁）

需要快速确认环境和编译链时，可把任一一键命令中的
`FLAGRAND_PERF_PROFILE=acceptance` 改为 `FLAGRAND_PERF_PROFILE=smoke`。Smoke 不得替代正式回归结果。
