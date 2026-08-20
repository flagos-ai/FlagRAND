#pragma once

#include <cstdint>
#include <memory>
#include <torch/torch.h>

namespace flagrand {

class Generator {
 public:
  explicit Generator(uint64_t seed = 0, uint64_t offset = 0) : seed_(seed), offset_(offset) {}

  uint64_t seed() const { return seed_; }
  uint64_t offset() const { return offset_; }
  void set_seed(uint64_t seed) {
    seed_ = seed;
    offset_ = 0;
  }
  void set_offset(uint64_t offset) { offset_ = offset; }
  void advance(uint64_t words) { offset_ += words; }

 private:
  uint64_t seed_;
  uint64_t offset_;
};

using GeneratorPtr = std::shared_ptr<Generator>;

GeneratorPtr create_generator(uint64_t seed = 0, uint64_t offset = 0);
at::Tensor generate(Generator& generator, const at::Tensor& out);
at::Tensor generate_uniform(Generator& generator, const at::Tensor& out);
at::Tensor generate_uniform_double(Generator& generator, const at::Tensor& out);
at::Tensor generate_normal(Generator& generator, const at::Tensor& out, double mean, double stddev);
at::Tensor generate_lognormal(Generator& generator, const at::Tensor& out, double mean, double stddev);

}  // namespace flagrand
