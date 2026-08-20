#include "flagrand/operators.h"

#include <cstdint>
#include <string>

#include "flagrand/backend_utils.h"
#include "flagrand/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flagrand {
namespace {

using triton_jit::TritonJITFunction;

constexpr int64_t kBlock = 512;
constexpr unsigned int kNumWarps = 4;
constexpr unsigned int kNumStages = 1;

unsigned int cdiv(uint64_t value, uint64_t divisor) {
  return static_cast<unsigned int>((value + divisor - 1) / divisor);
}

const TritonJITFunction& kernel(const char* name) {
  return TritonJITFunction::get_instance(utils::get_triton_source_path().string(), name);
}

void check_output(const at::Tensor& out, at::ScalarType dtype, bool even) {
  TORCH_CHECK(out.is_cuda(), "FlagRAND C++ wrapper requires an IX/CUDA-compatible tensor");
  TORCH_CHECK(out.is_contiguous(), "FlagRAND C++ wrapper requires contiguous output");
  TORCH_CHECK(out.scalar_type() == dtype, "FlagRAND C++ wrapper received the wrong output dtype");
  if (even) {
    TORCH_CHECK(out.numel() % 2 == 0, "normal and log-normal generation require an even element count");
  }
}

void launch_raw(const at::Tensor& out, uint64_t seed, uint64_t offset) {
  const uint64_t n = static_cast<uint64_t>(out.numel());
  if (n == 0) {
    return;
  }
  c10::DeviceGuard guard(out.device());
  const auto stream = backend::get_raw_stream(backend::get_current_stream(out.device()));
  const uint64_t base_counter = offset / 4;
  if (offset % 4 == 0 && n % 4 == 0) {
    const uint64_t counters = n / 4;
    kernel("raw_aligned_kernel")(
        stream, cdiv(counters, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, base_counter, counters, kBlock);
  } else {
    const uint64_t groups = (n + 3) / 4;
    kernel("raw_generic_kernel")(
        stream, cdiv(groups, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, base_counter, n, groups, static_cast<int64_t>(offset % 4), kBlock);
  }
}

void launch_uniform_f32(const at::Tensor& out, uint64_t seed, uint64_t offset) {
  const uint64_t n = static_cast<uint64_t>(out.numel());
  if (n == 0) {
    return;
  }
  c10::DeviceGuard guard(out.device());
  const auto stream = backend::get_raw_stream(backend::get_current_stream(out.device()));
  if (offset % 4 == 0 && n % 4 == 0) {
    const uint64_t counters = n / 4;
    kernel("uniform_f32_aligned_kernel")(
        stream, cdiv(counters, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, offset / 4, counters, kBlock);
  } else {
    kernel("uniform_f32_generic_kernel")(
        stream, cdiv(n, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, offset, n, kBlock);
  }
}

void launch_uniform_f64(const at::Tensor& out, uint64_t seed, uint64_t offset) {
  const uint64_t n = static_cast<uint64_t>(out.numel());
  if (n == 0) {
    return;
  }
  c10::DeviceGuard guard(out.device());
  const auto stream = backend::get_raw_stream(backend::get_current_stream(out.device()));
  if (offset % 4 == 0 && n % 2 == 0) {
    const uint64_t counters = n / 2;
    kernel("uniform_f64_aligned_kernel")(
        stream, cdiv(counters, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, offset / 4, counters, kBlock);
  } else {
    const uint64_t groups = (n + 1) / 2;
    kernel("uniform_f64_generic_kernel")(
        stream, cdiv(groups, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, offset / 4, n, groups, static_cast<int64_t>(offset % 4), kBlock);
  }
}

void launch_normal_f32(
    const at::Tensor& out,
    uint64_t seed,
    uint64_t offset,
    float mean,
    float stddev,
    bool lognormal) {
  const uint64_t n = static_cast<uint64_t>(out.numel());
  if (n == 0) {
    return;
  }
  c10::DeviceGuard guard(out.device());
  const auto stream = backend::get_raw_stream(backend::get_current_stream(out.device()));
  if (offset % 4 == 0 && n % 4 == 0) {
    const uint64_t counters = n / 4;
    kernel("normal_f32_aligned_kernel")(
        stream, cdiv(counters, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, offset / 4, counters, mean, stddev, lognormal, kBlock);
  } else if (offset % 2 == 0) {
    const uint64_t pairs = n / 2;
    kernel("normal_f32_pair_kernel")(
        stream, cdiv(pairs, kBlock), 1, 1, kNumWarps, kNumStages,
        out, seed, offset, pairs, mean, stddev, lognormal, kBlock);
  } else {
    const at::Tensor raw = at::empty({static_cast<int64_t>(n)}, out.options().dtype(at::kInt));
    launch_raw(raw, seed, offset);
    const uint64_t pairs = n / 2;
    kernel("normal_f32_from_raw_kernel")(
        stream, cdiv(pairs, kBlock), 1, 1, kNumWarps, kNumStages,
        out, raw, pairs, mean, stddev, lognormal, kBlock);
  }
}

}  // namespace

GeneratorPtr create_generator(uint64_t seed, uint64_t offset) {
  return std::make_shared<Generator>(seed, offset);
}

at::Tensor generate(Generator& generator, const at::Tensor& out) {
  check_output(out, at::kInt, false);
  launch_raw(out, generator.seed(), generator.offset());
  generator.advance(static_cast<uint64_t>(out.numel()));
  return out;
}

at::Tensor generate_uniform(Generator& generator, const at::Tensor& out) {
  check_output(out, at::kFloat, false);
  launch_uniform_f32(out, generator.seed(), generator.offset());
  generator.advance(static_cast<uint64_t>(out.numel()));
  return out;
}

at::Tensor generate_uniform_double(Generator& generator, const at::Tensor& out) {
  check_output(out, at::kDouble, false);
  launch_uniform_f64(out, generator.seed(), generator.offset());
  generator.advance(2 * static_cast<uint64_t>(out.numel()));
  return out;
}

at::Tensor generate_normal(Generator& generator, const at::Tensor& out, double mean, double stddev) {
  check_output(out, at::kFloat, true);
  launch_normal_f32(out, generator.seed(), generator.offset(), static_cast<float>(mean), static_cast<float>(stddev), false);
  generator.advance(static_cast<uint64_t>(out.numel()));
  return out;
}

at::Tensor generate_lognormal(Generator& generator, const at::Tensor& out, double mean, double stddev) {
  check_output(out, at::kFloat, true);
  launch_normal_f32(out, generator.seed(), generator.offset(), static_cast<float>(mean), static_cast<float>(stddev), true);
  generator.advance(static_cast<uint64_t>(out.numel()));
  return out;
}

}  // namespace flagrand
