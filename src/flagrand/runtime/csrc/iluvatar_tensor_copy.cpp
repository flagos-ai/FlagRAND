#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

// Python owns the raw/derived source and orders stream transitions before this
// call. Output addresses are resolved for each actual Tensor, including views.
void copy_tensor(const at::Tensor& out, const at::Tensor& source,
                 std::int64_t source_elements, std::int64_t count) {
  const auto stream = c10::cuda::getCurrentCUDAStream(out.get_device());
  const auto itemsize = source.element_size();
  const auto* input = static_cast<const char*>(source.data_ptr()) + source_elements * itemsize;
  const auto status = cudaMemcpyAsync(out.data_ptr(), input, count * itemsize,
                                      cudaMemcpyDeviceToDevice, stream.stream());
  if (status != cudaSuccess) {
    throw std::runtime_error("cudaMemcpyAsync returned " + std::to_string(static_cast<int>(status)));
  }
  out.record_stream(stream.unwrap());
}

namespace py = pybind11;

bool parameter_bits(py::handle object, bool binary64, std::uint64_t& result) {
  if (!PyFloat_CheckExact(object.ptr())) return false;
  const double value = PyFloat_AS_DOUBLE(object.ptr());
  if (binary64) {
    std::memcpy(&result, &value, sizeof(value));
    return true;
  }
  const float narrowed = static_cast<float>(value);
  if (std::isfinite(value) && !std::isfinite(narrowed)) return false;
  std::uint32_t bits;
  std::memcpy(&bits, &narrowed, sizeof(bits));
  result = bits;
  return true;
}

class CorexCachedOutput {
 public:
  CorexCachedOutput(py::object raw_object, const at::Tensor& source,
                    py::object entry, py::object seed, std::int64_t raw_start,
                    py::object mode, std::int64_t stride, std::int64_t residue,
                    std::int64_t outputs_per_unit, std::uint64_t mean_bits,
                    std::uint64_t stddev_bits, std::uint64_t lambda_bits)
      : raw_object_(std::move(raw_object)), raw_(raw_object_.cast<at::Tensor>()),
        source_(source), entry_(std::move(entry)), seed_(std::move(seed)), mode_(std::move(mode)),
        cuda_stream_(c10::cuda::getCurrentCUDAStream(source_.get_device())), stream_(cuda_stream_.unwrap()),
        raw_start_(raw_start), raw_end_(raw_start + raw_.numel()), stride_(stride), residue_(residue),
        outputs_per_unit_(outputs_per_unit), mean_bits_(mean_bits), stddev_bits_(stddev_bits),
        lambda_bits_(lambda_bits), dtype_(source_.scalar_type()), pair_(outputs_per_unit == 2),
        poisson_(PyUnicode_CompareWithASCIIString(mode_.ptr(), "poisson") == 0),
        source_pointer_(static_cast<const char*>(source_.data_ptr())), itemsize_(source_.element_size()) {}

  std::int64_t try_copy(const at::Tensor& out, py::handle current_raw,
                        py::handle current_entry, py::handle seed, std::int64_t offset,
                        py::handle mode, py::handle mean, py::handle stddev, py::handle lam) const {
    if (current_raw.ptr() != raw_object_.ptr() || current_entry.ptr() != entry_.ptr()) return 0;
    if (!PyLong_CheckExact(seed.ptr())) return 0;
    if (seed.ptr() != seed_.ptr() && PyObject_RichCompareBool(seed.ptr(), seed_.ptr(), Py_EQ) != 1) return 0;
    if (mode.ptr() != mode_.ptr() && (!PyUnicode_CheckExact(mode.ptr()) ||
        PyUnicode_Compare(mode.ptr(), mode_.ptr()) != 0)) return 0;
    if (out.scalar_type() != dtype_ || out.device() != source_.device()) return 0;
    const auto n = out.numel();
    if (n <= 0 || n > (poisson_ ? 1048577 : 65536)) return 0;
    const auto words = stride_ * (pair_ ? (n + 1) / 2 : n);
    if (offset < raw_start_ || offset > raw_end_ - words) return 0;
    const auto relative = offset - raw_start_;
    if (relative % stride_ != residue_) return 0;
    if (pair_) {
      std::uint64_t mean_bits, stddev_bits;
      if (!parameter_bits(mean, false, mean_bits) || !parameter_bits(stddev, false, stddev_bits) ||
          mean_bits != mean_bits_ || stddev_bits != stddev_bits_) return 0;
    }
    if (poisson_) {
      std::uint64_t lambda_bits;
      if (!parameter_bits(lam, true, lambda_bits) || lambda_bits != lambda_bits_) return 0;
    }
    // Full stream id/device/type comes from the installed backend TLS.
    if (c10::cuda::getCurrentCUDAStream(out.get_device()).unwrap() != stream_) return 0;
    const auto source_elements = outputs_per_unit_ * ((relative - residue_) / stride_);
    {
      py::gil_scoped_release release;
      const auto status = cudaMemcpyAsync(out.data_ptr(), source_pointer_ + source_elements * itemsize_,
                                          n * itemsize_, cudaMemcpyDeviceToDevice, cuda_stream_.stream());
      if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync returned " + std::to_string(static_cast<int>(status)));
      }
      out.record_stream(stream_);
    }
    return words;
  }

  py::dict metadata() const {
    py::dict result;
    result["stream_id"] = stream_.id();
    result["device_index"] = stream_.device_index();
    result["device_type"] = static_cast<int>(stream_.device_type());
    result["raw_handle"] = reinterpret_cast<std::uintptr_t>(cuda_stream_.stream());
    result["raw_start"] = raw_start_;
    result["raw_end"] = raw_end_;
    result["stride"] = stride_;
    result["residue"] = residue_;
    result["outputs_per_unit"] = outputs_per_unit_;
    result["source_numel"] = source_.numel();
    result["mean_bits"] = mean_bits_;
    result["stddev_bits"] = stddev_bits_;
    result["lambda_bits"] = lambda_bits_;
    return result;
  }

 private:
  py::object raw_object_;
  at::Tensor raw_;
  at::Tensor source_;
  py::object entry_;
  py::object seed_;
  py::object mode_;
  c10::cuda::CUDAStream cuda_stream_;
  c10::Stream stream_;
  std::int64_t raw_start_, raw_end_, stride_, residue_, outputs_per_unit_;
  std::uint64_t mean_bits_, stddev_bits_, lambda_bits_;
  at::ScalarType dtype_;
  bool pair_, poisson_;
  const char* source_pointer_;
  std::size_t itemsize_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("copy_tensor", &copy_tensor, pybind11::call_guard<pybind11::gil_scoped_release>());
  py::class_<CorexCachedOutput>(module, "CachedOutput")
      .def(py::init<py::object, const at::Tensor&, py::object, py::object, std::int64_t,
                    py::object, std::int64_t, std::int64_t, std::int64_t,
                    std::uint64_t, std::uint64_t, std::uint64_t>())
      .def("try_copy", &CorexCachedOutput::try_copy)
      .def("metadata", &CorexCachedOutput::metadata);
  // Diagnostic only; both scheduling and allocator bookkeeping above use the
  // actual backend TLS stream, without constructing a stream from a pointer.
  module.def("current_stream_identity", [](c10::DeviceIndex device) {
    const auto current = c10::cuda::getCurrentCUDAStream(device);
    const auto stream = current.unwrap();
    pybind11::dict result;
    result["stream_id"] = stream.id();
    result["device_index"] = stream.device_index();
    result["device_type"] = static_cast<int>(stream.device_type());
    result["raw_handle"] = reinterpret_cast<std::uintptr_t>(current.stream());
    return result;
  });
}
