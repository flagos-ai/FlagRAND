#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <torch/torch.h>

namespace flagrand::backend {

using StreamType = c10::cuda::CUDAStream;
using RawStreamType = CUstream;

inline StreamType get_current_stream(const at::Device& device) {
  return c10::cuda::getCurrentCUDAStream(device.index());
}

inline RawStreamType get_raw_stream(const StreamType& stream) {
  return stream.stream();
}

}  // namespace flagrand::backend
