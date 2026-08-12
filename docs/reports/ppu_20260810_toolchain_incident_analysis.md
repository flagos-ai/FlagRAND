# PPU Triton 3.6 编译链兼容问题技术复盘

## 1. 结论摘要

2026-08-10 下午 PPU 回归失败的主因，不是 PPU-ZW810E 硬件故障、FlagRAND Philox 算法错误、Ascend 910C 代码污染，也不是普通的 Triton 缓存失效，而是当前节点两段编译链之间存在一处确定的文本方言不兼容：

- FlagTree `0.6.0+ppu3.6` 所带 Triton 3.6 PPU backend 在 LLVM inline assembly 中生成 `ppu.st.global.b32`、`ppu.ex2.approx.ftz.f32` 等带 `ppu.` namespace 的 opcode；
- 节点 PPU SDK `2.0.0-715aa1` 中的 `/usr/local/PPU_SDK/bin/ppu-llc` 只接受 `st.global.b32`、`ex2.approx.ftz.f32` 等不带 namespace 的 opcode；
- `ppu-llc` 因此在读取 Triton 生成的 `.tix.trans` 时直接发生词法/语法解析失败，kernel 无法生成最终二进制。

此外，FlagTree PPU backend 的失败处理分支还引用了未初始化的 `log_file`。该问题不是编译失败的起因，却用一个二次 `NameError` 覆盖了 `ppu-llc` 的真实 stderr，使最初看到的错误具有很强的误导性。

最终修复是在 FlagRAND 进程内安装一个仅对 PPU backend 生效的 compiler compatibility proxy：拦截发往 `ppu-llc` 的调用，只复制并修改 `.tix.trans` 临时输入，将 `ppu.` namespace 去除，再把临时副本交给原始 `ppu-llc`；同时初始化 backend 异常分支需要的日志路径。整个方案不修改系统 Triton、PPU SDK、原始编译输入或随机数算法。

修复后的实机结果为：

- PPU 七 API preflight：通过；
- PPU 3.6 cached-launch ABI：直接复用编译二进制与标准 JIT 输出 bitwise equal，最大差值为 0；
- 正式 acceptance：324 条记录、0 failure、7/7 API 通过 `0.95x` 单 API 聚合门禁。

## 2. 问题的准确分层

下午遇到的现象实际上由三个层次组成，必须分别看待。

| 层次 | 问题 | 性质 | 是否为首次编译失败的根因 |
| --- | --- | --- | --- |
| 编译输入方言 | Triton PPU backend 输出 `ppu.*`，`ppu-llc` 仅接受无前缀 opcode | 当前 FlagTree 与 PPU SDK 组合的接口不兼容 | 是 |
| 编译异常处理 | `compiler.py` 在失败分支使用未初始化的 `log_file` | 厂商 backend 的错误处理缺陷 | 否，但会遮蔽主因 |
| 运行时策略 | target 从历史 `cuda` 变为 `ppu`，原有 cached-launch/portable-math 策略需要识别新 backend | 版本迁移后的运行时适配 | 否，属于编译打通后的性能与兼容适配 |

因此，“PPU 环境坏了”并不准确。更准确的结论是：节点当前安装的软件栈可以工作，但 Triton 3.6 PPU backend 与配套 `ppu-llc` 在 inline-assembly opcode 命名规则上没有对齐，FlagRAND 需要在这条边界上做兼容。

## 3. 环境变化为什么重要

此前已经成功验收的 PPU 环境和下午使用的环境，并不是同一条 Triton 编译路径。

| 项目 | 2026-07-21 成功环境 | 2026-08-10 当前环境 |
| --- | --- | --- |
| 设备 | PPU-ZW810E | PPU-ZW810E |
| torch | 2.9.0 | 2.9.0 |
| Triton | 3.5.0 | 3.6.0 |
| Triton target | `GPUTarget(backend='cuda', arch=80, warp_size=32)` | `GPUTarget(backend='ppu', arch=80, warp_size=32)` |
| backend 路径 | CUDA compatibility target | first-class PPU backend |
| FlagTree distribution | 旧环境未走当前 PPU 3.6 backend | `flagtree 0.6.0+ppu3.6` |
| PPU SDK | PPU 环境 | `2.0.0-715aa1` |

设备名称、torch 版本和 Python API 表面上没有明显变化，但 target backend 已从 `cuda` 切换到 `ppu`。这意味着：

1. kernel 不再经过旧的 CUDA compatibility 编译路径，而是进入 `/usr/local/lib/python3.12/site-packages/triton/backends/ppu/compiler.py`；
2. PPU backend 开始直接生成 PPU 方言的中间输入；
3. backend 名称变化也会影响 FlagRAND 对 cached binary launch 和 portable FP32 math 的策略判断。

所以，旧版本源码在当前节点上失败，并不能单独证明“代码回归”或“节点整体损坏”。旧源码仍然会进入当前的 PPU 3.6 compiler backend，最终撞上同一个 `ppu-llc` 方言边界。

## 4. 编译链中真正失败的位置

当前调用链可以简化为：

```text
FlagRAND public API
  -> Triton JIT
  -> FlagTree PPU backend
  -> 生成 LLVM/TIX 临时输入 *.tix.trans
  -> /usr/local/PPU_SDK/bin/ppu-llc
  -> PPU kernel object / hgbin
  -> CompiledKernel launch
```

失败发生在 `.tix.trans -> ppu-llc` 这一步，尚未进入 PPU 设备执行阶段。捕获到的实际命令形如：

```text
/usr/local/PPU_SDK/bin/ppu-llc ... \
  /tmp/tmpaperrair.tix.trans \
  -o /tmp/tmpaperrair.tix.trans.o
```

生成文件中的 store inline assembly 为：

```llvm
tail call void asm sideeffect "@$2 ppu.st.global.b32 [ $1 + 0 ], { $0 };", ...
```

`ppu-llc` 返回的第一组真实错误是：

```text
line 1:7 token recognition error at: '.st.'
line 1:4 mismatched input 'ppu' expecting {... OPCODE_ST ...}
```

在去除 store 前缀后继续编译 normal/lognormal 路径，又暴露了同类错误：

```text
line 1:3 token recognition error at: '.ex2'
```

对应输入中的 opcode 是：

```llvm
tail call float asm "ppu.ex2.approx.ftz.f32", ...
```

这两组错误共同说明，问题不是某一条 store 指令的特殊写法，而是同一类 opcode namespace 协议不一致：生成端把 `ppu.` 当作方言限定符输出，消费端 parser 却把第一个 token 直接当作 opcode，并不认识该限定符。

## 5. 为什么最初只看到 `NameError`

系统 PPU backend 的核心异常路径为：

```python
try:
    subprocess.run(ppullc_cmd, shell=True, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as error:
    ...
    with open(log_file, "a") as output:
        ...
```

正常路径只在启用特定 compile-log 选项时初始化或使用相关日志变量；失败路径却无条件打开 `log_file`。因此发生了以下覆盖链：

```text
ppu-llc 解析 ppu.st.global.b32 失败
  -> subprocess.run 抛出 CalledProcessError
  -> compiler.py 进入 except
  -> except 访问未初始化的 log_file
  -> 抛出 NameError
  -> 用户最终只看到 NameError
```

这解释了为什么最初的报错看起来像一个普通 Python 变量问题。给 `log_file` 一个合法路径只能恢复诊断信息，不能让 kernel 编译成功；真正需要解决的仍是 `ppu-llc` 输入方言不匹配。

## 6. 排查中如何排除错误方向

### 6.1 排除 Ascend 910C 代码污染

融合包最初令人怀疑 PPU 与 910C 路径发生了相互污染。实际检查结果为：

- PPU 进程加载后，`sys.modules` 中不存在 `flagrand.ascend` 或其子模块；
- active target 明确为 `backend='ppu'`；
- PPU/Ascend source boundary 静态检查通过；
- 在隔离的 PPU v3 目录和全新 Triton cache 下仍复现同一错误。

因此 910C 实现没有进入 PPU 编译路径，也没有造成这次失败。

### 6.2 排除 Philox 算法和分布变换

为判断问题是在随机数计算还是更底层，依次做了以下替换：

- 将手写 Philox 路径替换为 Triton 内建 `tl.philox`；
- 恢复旧的 PPU vector-store 写法；
- 构造仅执行一次全局 store、不包含 Philox round、不包含 uniform/normal 变换的最小 kernel。

三类实验都在相同的 `ppu.st.global.b32` 处失败。尤其 store-only kernel 不包含随机数算法，证明故障位于编译器边界，而不是 Philox、Box-Muller、FP32 数学或 output offset 逻辑。

### 6.3 排除普通 cache 问题

旧源码分别使用原有默认 cache 和全新空 cache，均进入相同失败链。全新 cache 会强制重新编译，因此可以确认：

- 失败不是坏 cache 条目导致；
- 旧 cache 即使偶尔能绕过编译，也只能隐藏兼容问题，不能作为交付修复；
- 正式修复必须保证 fresh-cache preflight 通过。

### 6.4 识别命令行与工作目录噪声

排查中还出现过 shell 粘贴损坏、未切入项目目录导致 pytest 找不到 `tests/accuracy`、日志目录不存在等错误。这些会造成额外失败，但与 `ppu-llc` parser 无关。它们通过固定项目路径和一键脚本解决，不属于本次 PPU 编译事故的根因。

## 7. 根因验证，而非仅凭猜测修补

### 7.1 最小输入 A/B 实验

对同一份 2.8 KiB store-only `.tix.trans` 做了严格对照：

```text
原始输入:     ppu.st.global.b32   -> ppu-llc RC=1
仅去除前缀:  st.global.b32       -> ppu-llc RC=0
```

归一化后的输出是有效 ELF object。实验只改变了 `ppu.` 前缀，其余 IR、编译参数、`ppu-llc` 可执行文件均未变化，因此可以建立直接因果关系。

### 7.2 复杂输入矩阵

随后对当时保留的 13 份 `.tix.trans` 做批量验证：

- 输入大小覆盖 2,819 B 到 36,017 B；
- 每份包含 1、4 或 16 个 `ppu.` token；
- 归一化后剩余前缀数均为 0；
- 13/13 输入经同一个 `ppu-llc` 编译成功，return code 全部为 0。

这一步证明方法不仅能修复最小 store probe，也覆盖完整 Philox 和分布 kernel 生成的复杂输入。

### 7.3 store 与 ex2 静态覆盖

初版兼容实验只解决了先遇到的 store 形式，编译继续推进后在 `ppu.ex2.approx.ftz.f32` 处再次失败。最终归一化同时验证：

```text
asm sideeffect "st.global.b32"
tail call float asm "ex2.approx.ftz.f32"
```

这说明修复目标应是 `ppu.` namespace 本身，而不是为 `st` 和 `ex2` 分别堆叠特例。

## 8. 最终实现

最终实现位于 [`src/flagrand/runtime/ppu_compiler.py`](../../src/flagrand/runtime/ppu_compiler.py)，安装入口位于 [`src/flagrand/runtime/backend.py`](../../src/flagrand/runtime/backend.py)。

### 8.1 只在 active PPU backend 安装

`current_backend()` 读取 Triton active target。只有 backend 等于 `ppu` 时才导入并安装兼容层。CUDA、Ascend NPU 和其他平台不会加载该逻辑，从入口上保证平台隔离。

### 8.2 在进程内拦截实际 compiler 调用

系统 backend 使用绝对路径 `/usr/local/PPU_SDK/bin/ppu-llc`，因此即便把自定义 wrapper 放在 `PATH` 首位，实际编译仍会绕过 wrapper。排查时已经观察到：

- `which ppu-llc` 指向项目 wrapper；
- 实际失败命令仍以 `/usr/local/PPU_SDK/bin/ppu-llc` 开头；
- wrapper activity log 为空。

最终方案不再依赖 `PATH`，而是代理 PPU compiler 模块使用的 `subprocess` 对象。代理只拦截可执行文件 basename 为 `ppu-llc` 的 `run()` 调用，其他 subprocess 行为全部转发给原对象。

### 8.3 只改临时编译输入

代理定位命令参数中真实存在且以 `.tix.trans` 结尾的输入文件，然后：

1. 读取原始编译输入；
2. 在进程私有 `TemporaryDirectory` 中创建副本；
3. 对副本执行 `source.replace("ppu.", "")`；
4. 仅把命令中的输入路径替换为副本路径；
5. 使用原始参数调用原始 `ppu-llc`；
6. 编译结束后自动清理临时目录。

该实现不修改：

- `/usr/local/PPU_SDK/bin/ppu-llc`；
- `/usr/local/lib/python3.12/site-packages/triton/backends/ppu/compiler.py` 文件本身；
- Triton cache 中的原始生成物；
- FlagRAND Philox 位流、分布算法或 public API。

兼容层可以通过 `FLAGRAND_PPU_LLC_COMPAT_LOG` 记录每次实际归一化的输入数量；`FLAGRAND_PPU_COMPILE_LOG` 可指定 backend 异常日志位置。

### 8.4 修复错误可观测性

安装兼容层时显式设置 `ppu_compiler.log_file`。若用户未配置 `FLAGRAND_PPU_COMPILE_LOG`，默认放入 `TRITON_CACHE_DIR/flagrand_ppu_compile.log`，并提前创建父目录。这样即使未来遇到其他 `ppu-llc` 错误，也会保留真正的 compiler stderr，不再被 `NameError` 覆盖。

## 9. 编译打通后的第二层适配：cached launch

方言修复解决“能不能编译”。要达到交付性能，还需要解决“当前 `backend='ppu'` 能不能安全复用编译二进制”。这不是同一个问题。

历史环境把 PPU 暴露为 `cuda` target，原有策略自然会启用 cached binary launch；当前环境变为 `ppu` 后，如果只按 backend 名称判断，可能回退到普通 JIT launch，造成不必要的 host-side launch 开销。

在启用 PPU cached path 前，进行了直接 ABI 探针：

- `CompiledKernel.run` 可调用，签名为 `(gridX, gridY, gridZ, stream, function, *args)`；
- device cache entry 长度为 5，第 5 个 slot 是可调用 binder；
- `compiled.function`、`compiled.packed_metadata`、`compiled.launch_metadata` 和 `_init_handles` 均可用；
- launch metadata 类型为 `LazyDict`；
- 相同输入下，标准 JIT launch 与直接 compiled launch 的 raw32 输出 bitwise equal，最大整数差为 0。

据此，[`src/flagrand/runtime/backend.py`](../../src/flagrand/runtime/backend.py) 将 `ppu` 纳入 cached binary launch 支持集合，同时让 PPU 继续使用 portable FP32 math。正式策略为：

```text
backend: ppu
cached binary launch: True
portable f32 math: True
```

[`src/flagrand/runtime/compiled_launcher.py`](../../src/flagrand/runtime/compiled_launcher.py) 在复用 specialization 时重新取得当前 stream，并按实机确认的 PPU 3.6 ABI 传入 function、packed metadata、launch metadata 和实际参数。这样既避免每次走完整 JIT launcher，又不把旧 CUDA ABI 假定强行套到 PPU 上。

## 10. 为什么没有采用其他“看起来更直接”的方案

### 10.1 不直接修改系统 Triton 或 PPU SDK

修改 site-packages 或 `/usr/local/PPU_SDK` 可以暂时跑通，但会污染共享环境、难以审计和回滚，也无法随项目交付。项目内兼容层可版本化、可测试、可随 tar 包迁移，风险边界更清晰。

### 10.2 不依赖 PATH wrapper

系统 backend 直接构造绝对 `ppu-llc` 路径，PATH wrapper 已通过 activity log 证明没有被调用。继续调整 PATH 不会解决问题。

### 10.3 不继续重写 Philox kernel

store-only probe 已经在零随机数运算条件下复现失败；继续修改乘法、位运算、store 组织或 normal 数学只会改变生成 IR 的形状，无法修复 compiler parser 的 token 协议。

### 10.4 不依赖旧 Triton cache

旧 cache 可能暂时跳过出错编译阶段，但不能保证新 shape、新 specialization、CI 或其他节点可运行。交付必须通过全新 cache 编译，这也是最终验证的一部分。

### 10.5 不简单宣布“厂商环境坏了”

硬件可见、torch 可用、PPU native baseline 可用，且去除 namespace 后原始 `ppu-llc` 可以生成有效 object。这些证据说明 PPU SDK 并非整体不可用；问题集中在当前两个软件组件的接口契约上。

## 11. 最终验证结果

### 11.1 编译与正确性

| 验证项 | 结果 |
| --- | --- |
| store-only 原始输入 | `ppu-llc` RC=1 |
| 同一输入去除 `ppu.` | `ppu-llc` RC=0，生成有效 ELF |
| 保留的复杂 `.tix.trans` 矩阵 | 13/13 编译成功 |
| store/ex2 namespace 静态检查 | 通过 |
| fresh-cache 七 API preflight | 通过 |
| compiled direct launch 与标准 JIT | bitwise equal，max diff=0 |

### 11.2 正式性能验收

环境：PPU-ZW810E、torch 2.9.0、FlagTree `0.6.0+ppu3.6`、Triton 3.6.0、PPU SDK `2.0.0-715aa1`，baseline 为 PPU native `acrand`。

| API | 单 API 平均加速比 |
| --- | ---: |
| raw32 | 1.104x |
| uniform_f32 | 0.987x |
| uniform_f64 | 1.175x |
| normal_f32 | 1.113x |
| normal_f64 | 1.029x |
| lognormal_f32 | 1.179x |
| lognormal_f64 | 1.033x |

最终 acceptance 为 324 records、0 failures，7/7 API 达到 `0.95x` 门禁。实机结果目录为：

```text
/home/flagrand_stage4_ppu_20260810_v3/FlagRAND-pr1/ppu_results/
ppu36_cached_acceptance.i8ElWP/20260810_143548_acceptance_full_philox
```

## 12. 事故定性

本次问题可以定性为：

> FlagTree Triton 3.6 PPU backend 与节点 PPU SDK `ppu-llc` 之间的 inline-assembly opcode namespace 兼容缺陷，并伴随 backend 异常处理路径的日志变量初始化缺陷。

以下判断均已被实验排除：

- 不是 PPU-ZW810E 硬件不支持 FlagRAND；
- 不是 PPU 环境整体损坏；
- 不是 FlagRAND Philox 或分布算法写错；
- 不是融合 Ascend 910C 后发生模块污染；
- 不是首次编译没有 cache 导致的正常慢启动；
- 不是删除或重建 cache 就能修复的问题。

## 13. 后续工程建议

1. PPU 环境指纹必须同时记录 FlagTree、Triton、target backend、PPU SDK 和 compiler 文件哈希，不能只记录 torch、设备名和 Python 版本。
2. CI 应保留 fresh-cache preflight。只有 cached run 成功不能证明编译链兼容。
3. compiler boundary 测试应至少覆盖 global store 和数学 intrinsic 两类 inline assembly；当前单测见 [`tests/accuracy/test_ppu_compiler.py`](../../tests/accuracy/test_ppu_compiler.py)。
4. `ppu-llc` 失败时应优先保存真实命令、return code、stdout、stderr 和对应 `.tix.trans`，避免高层异常覆盖原始证据。
5. compatibility proxy 必须继续限制在 active `ppu` backend 和 `.tix.trans` 输入，避免影响 CUDA、Ascend 910C 或普通 subprocess。
6. 后续若厂商 backend 与 SDK 已统一 namespace 规则，应使用原始输入 A/B 测试确认，再考虑移除兼容层；在此之前不应依据版本号猜测兼容问题已经消失。
7. cached-launch 支持必须经过 ABI 与 bitwise replay 验证，不能仅因 backend 名称相似而默认开启。

## 14. 最终结论

下午的 PPU 故障是一次典型的编译工具链接口漂移：硬件、随机数算法和 native baseline 都正常，真正断裂的是 Triton 3.6 PPU backend 输出的 opcode 表达方式与当前 `ppu-llc` parser 的输入契约。`log_file` 的二次异常使问题最初看起来远比实际混乱，而旧环境的 `cuda` target 又掩盖了“软件路径已发生切换”这一关键事实。

通过最小 kernel、fresh cache、平台导入隔离、原始 compiler stderr 捕获、同源输入 A/B 编译和 13 份复杂输入矩阵，最终把问题收敛到一个可重复、可验证的 namespace 差异。项目内临时输入归一化既解决了编译兼容，又保持系统环境和算法实现不变；在进一步完成 PPU 3.6 cached-launch ABI 验证后，七个 API 的正确性和性能验收全部恢复。
