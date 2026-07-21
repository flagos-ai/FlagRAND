# FlagRAND PPU 性能与正确性总结

## 结论

本次在 PPU-ZW810E 上完成了 Philox 路径下四类随机数算子族、七个端到端 API 的迁移、性能验证和正确性核验：

- Raw
- Uniform
- Normal
- LogNormal

其中 Raw、Uniform 支持任意长度；Normal、LogNormal 基于 Box-Muller，一次生成两个值，因此输出长度要求为偶数。

本次验收口径采用固定 shape 集合逐 API 报告：每个端到端 API 先对自身多个 shape 的加速比求平均，对标 baseline 为 PPU SDK native `acrand`。

2026-07-21 `acceptance_wide` 实测结果：

```text
Matrix records: 324 records, 0 failures
Acceptance status: pass, 7/7 APIs in target range
```

| API | API 加速比 |
| --- | ---: |
| Raw32 | 1.112x |
| Uniform FP32 | 1.016x |
| Uniform FP64 | 1.224x |
| Normal FP32 | 1.110x |
| Normal FP64 | 1.033x |
| LogNormal FP32 | 1.183x |
| LogNormal FP64 | 1.033x |

## 计算口径

单个 shape 的加速比：

```text
speedup(shape) = PPU acrand total ms / FlagRAND total ms
```

每个 API 的加速比：

```text
api_speedup = mean(speedup(shape_1), speedup(shape_2), ...)
```



本次验收纳入的 shape：

| API | 纳入 shape |
| --- | --- |
| Raw32 | `128,256,1023,1024,1025,4095,4096,4097,16384,65535,65536,65537,262144,1048575,1048576,1048577` |
| Uniform FP32 | `128,256,1023,1024,1025,4095,4096,4097,16384,65535,65536,65537,262144,1048575,1048576,1048577` |
| Uniform FP64 | `128,256,1023,1024,1025,4095,4096,4097,16384,65535,65536,65537,262144,1048575,1048576,1048577` |
| Normal FP32 | `126,128,130,1022,1024,1026,4094,4096,4098,16384,65534,65536,65538,262144,1048574,1048576,1048578` |
| Normal FP64 | `126,128,1024,1026,4096,4098,16384,65534,65536,65538,262144` |
| LogNormal FP32 | `126,128,130,1022,1024,1026,4094,4096,4098,16384,65534,65536,65538,262144,1048574,1048576,1048578` |
| LogNormal FP64 | `18,126,128,130,1022,1024,1026,4094,4096,4098,16384,65534,65536,65538,262144` |

## 对标 API

本次性能对标采用 PPU SDK 中的 native `libacrand.so`。PPU SDK 同时提供 `libcurand.so` 兼容层，实测 `ppu_curand` 与 `ppu_acrand` 结果基本一致；文档主表使用 `ppu_acrand` 作为正式 baseline。

正式性能测试使用默认 fast/cached 路径，不设置 `FLAGRAND_FORCE_PORTABLE=1`。该变量仅用于兼容性排障；开启后会禁用 cached binary launcher，并可能让部分 f32 normal/lognormal 使用 portable math，因此不作为正式性能口径。


| 算子 | FlagRAND 端到端 API | PPU baseline API |
| --- | --- | --- |
| Raw32 | `curand.generate` | `acrandGenerate` |
| Uniform FP32 | `curand.generate_uniform` | `acrandGenerateUniform` |
| Uniform FP64 | `curand.generate_uniform_double` | `acrandGenerateUniformDouble` |
| Normal FP32 | `curand.generate_normal` | `acrandGenerateNormal` |
| Normal FP64 | `curand.generate_normal_double` | `acrandGenerateNormalDouble` |
| LogNormal FP32 | `curand.generate_lognormal` | `acrandGenerateLogNormal` |
| LogNormal FP64 | `curand.generate_lognormal_double` | `acrandGenerateLogNormalDouble` |

## 验收性能结果

完整结果目录：

```text
ppu_results/acceptance_wide/20260721_134131_acceptance_full_philox
```


## 代表性明细

下表只列代表性 shape，完整验收结果以上文逐 API shape 加速比为准。

| API | N | 类型 | FlagRAND total ms | PPU acrand total ms | 加速比 |
| --- | ---: | --- | ---: | ---: | ---: |
| Raw32 | 4097 | 奇数长度 | 0.0165 | 0.0096 | 0.585x |
| Raw32 | 262144 | 中等 shape | 0.0128 | 0.0138 | 1.077x |
| Raw32 | 1048576 | 1M 级对齐长度 | 0.0130 | 0.0459 | 3.538x |
| Uniform FP32 | 262144 | 中等 shape | 0.0137 | 0.0129 | 0.941x |
| Uniform FP32 | 1048576 | 1M 级对齐长度 | 0.0139 | 0.0424 | 3.041x |
| Uniform FP64 | 262144 | 中等 shape | 0.0125 | 0.0134 | 1.072x |
| Uniform FP64 | 1048577 | 1M 级奇数长度 | 0.0153 | 0.0503 | 3.282x |
| Normal FP32 | 4098 | 非 4 倍数偶数 | 0.0182 | 0.0103 | 0.567x |
| Normal FP32 | 262144 | 中等 shape | 0.0156 | 0.0155 | 0.994x |
| Normal FP32 | 1048576 | 1M 级对齐长度 | 0.0159 | 0.0543 | 3.416x |
| Normal FP64 | 65538 | 非 4 倍数偶数 | 0.0163 | 0.0161 | 0.990x |
| Normal FP64 | 262144 | 中等 shape | 0.0145 | 0.0519 | 3.587x |
| LogNormal FP32 | 4098 | 非 4 倍数偶数 | 0.0182 | 0.0103 | 0.566x |
| LogNormal FP32 | 262144 | 中等 shape | 0.0158 | 0.0172 | 1.091x |
| LogNormal FP32 | 1048576 | 1M 级对齐长度 | 0.0158 | 0.0615 | 3.900x |
| LogNormal FP64 | 65538 | 非 4 倍数偶数 | 0.0157 | 0.0193 | 1.230x |
| LogNormal FP64 | 262144 | 中等 shape | 0.0145 | 0.0631 | 4.342x |

## 正确性核验

本次正确性核验采用结果导向，不依赖单一 oracle。

| 验证项 | 方法 | 结果 |
| --- | --- | --- |
| 预检查 | `ppu_flagrand_matrix.py --preflight-only`，覆盖 golden vector 和 bulk/split | pass |
| 回归测试 | `tests/test_api.py`、`tests/test_philox_cuda.py` | pass |
| 序列语义 | 相同 seed/offset 可复现，不同 seed/offset 输出隔离 | pass |
| 分段一致性 | 一次 bulk 生成与多次 split 生成结果一致 | pass |
| offset 泛用性 | 验证非 4 倍数 offset；修复 raw generic lane 顺序问题后通过 | pass |
| 长度泛用性 | raw/uniform 覆盖奇数长度；normal/lognormal 覆盖非 4 倍数偶数长度 | pass |
| 边界条件 | 零长度不推进 offset，大 offset 可复现 | pass |
| 分布统计 | Uniform/Raw 检查范围、均值、方差、KS、chi-square；Normal/LogNormal 检查均值、标准差、偏度、峰度、KS | pass |
| 独立性 | autocorrelation、runs test、serial test | pass |

PPU SDK 的 `acrand/libcurand` 与当前 Triton Philox 输出未做到 bitwise 对齐，PPU SDK 仅作为端到端性能 baseline；正确性依据来自 golden vector、序列语义、边界行为和统计检验。
