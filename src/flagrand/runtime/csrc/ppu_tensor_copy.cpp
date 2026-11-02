#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <hggc_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

// The installed Torch caster preserves the Python Stream's id/device/type.
// This accepts the same contiguous tensor windows as _copy_tensor; validation
// and source/cache ownership remain in the existing public implementation.
void copy_tensor(const at::Tensor& out, const at::Tensor& source,
                 std::int64_t source_elements, std::int64_t count,
                 c10::Stream stream) {
  const auto itemsize = source.element_size();
  const auto* input = static_cast<const char*>(source.data_ptr()) + source_elements * itemsize;
  const auto runtime_stream = c10::cuda::CUDAStream(stream).stream();
  const auto status = hggcMemcpyAsync(
      out.data_ptr(), input, count * itemsize, hggcMemcpyDeviceToDevice,
      reinterpret_cast<hggcStream_t>(runtime_stream));
  if (status) {
    throw std::runtime_error("hggcMemcpyAsync returned " + std::to_string(static_cast<int>(status)));
  }
  // Keep the Tensor method and installed ATen backend dispatch. A C++ argument
  // reference lasting only until enqueue returns is not asynchronous ownership.
  out.record_stream(stream);
}

pybind11::dict stream_identity(c10::Stream stream) {
  const auto data = stream.pack3();
  pybind11::dict result;
  result["stream_id"] = data.stream_id;
  result["device_index"] = data.device_index;
  result["device_type"] = static_cast<int>(data.device_type);
  result["device"] = stream.device().str();
  result["raw_handle"] = reinterpret_cast<std::uintptr_t>(c10::cuda::CUDAStream(stream).stream());
  return result;
}

namespace py = pybind11;

// Python struct.pack('<f') also rejects a finite input that overflows float.
// Non-float Python objects remain on the original Python conversion path.
bool float_bits(py::handle value, bool f64, std::uint64_t& result) {
  if (!PyFloat_CheckExact(value.ptr())) return false;
  const double number = PyFloat_AS_DOUBLE(value.ptr());
  if (f64) {
    std::memcpy(&result, &number, sizeof(number));
    return true;
  }
  const float narrowed = static_cast<float>(number);
  if (std::isfinite(number) && !std::isfinite(narrowed)) return false;
  std::uint32_t bits;
  std::memcpy(&bits, &narrowed, sizeof(bits));
  result = bits;
  return true;
}

py::dict stream_metadata(c10::Stream stream) {
  const auto data = stream.pack3();
  py::dict result;
  result["stream_id"] = data.stream_id;
  result["device_index"] = data.device_index;
  result["device_type"] = static_cast<int>(data.device_type);
  result["device"] = stream.device().str();
  result["raw_handle"] = reinterpret_cast<std::uintptr_t>(c10::cuda::CUDAStream(stream).stream());
  return result;
}

class CachedOutput {
 public:
  CachedOutput(py::object raw_object, const at::Tensor& source,
               py::object entry, py::object stream_object, py::object seed,
               std::int64_t raw_start, py::object mode, std::int64_t stride,
               std::int64_t residue, std::int64_t outputs_per_unit,
               std::uint64_t mean_bits, std::uint64_t stddev_bits)
      : raw_object_(std::move(raw_object)), raw_(raw_object_.cast<at::Tensor>()),
        source_(source), entry_(std::move(entry)), stream_object_(std::move(stream_object)),
        stream_(stream_object_.cast<c10::Stream>()), seed_(std::move(seed)), mode_(std::move(mode)),
        raw_start_(raw_start), raw_end_(raw_start + raw_.numel()), stride_(stride),
        residue_(residue), outputs_per_unit_(outputs_per_unit),
        mean_bits_(mean_bits), stddev_bits_(stddev_bits), dtype_(source_.scalar_type()),
        pair_(outputs_per_unit == 2), f64_(source_.scalar_type() == at::ScalarType::Double),
        source_pointer_(static_cast<const char*>(source_.data_ptr())), itemsize_(source_.element_size()),
        runtime_stream_(reinterpret_cast<hggcStream_t>(c10::cuda::CUDAStream(stream_).stream())) {}

  std::int64_t try_copy(const at::Tensor& out, py::handle current_raw,
                       py::handle current_entry, py::handle last_stream,
                       py::handle seed, std::int64_t offset, py::handle mode,
                       py::handle mean, py::handle stddev) const {
    if (current_raw.ptr() != raw_object_.ptr() || current_entry.ptr() != entry_.ptr() ||
        last_stream.ptr() != stream_object_.ptr()) return 0;
    if (!PyLong_CheckExact(seed.ptr())) return 0;
    if (seed.ptr() != seed_.ptr() && PyObject_RichCompareBool(seed.ptr(), seed_.ptr(), Py_EQ) != 1) return 0;
    if (mode.ptr() != mode_.ptr() && (!PyUnicode_CheckExact(mode.ptr()) ||
        PyUnicode_Compare(mode.ptr(), mode_.ptr()) != 0)) return 0;
    if (out.scalar_type() != dtype_ || out.device() != source_.device()) return 0;
    const auto n = out.numel();
    if (n <= 0 || n > 65536) return 0;
    const auto words = stride_ * (pair_ ? (n + 1) / 2 : n);
    if (offset < raw_start_ || offset > raw_end_ - words) return 0;
    const auto relative = offset - raw_start_;
    if (relative % stride_ != residue_) return 0;
    if (pair_) {
      std::uint64_t mean_bits, stddev_bits;
      if (!float_bits(mean, f64_, mean_bits) || !float_bits(stddev, f64_, stddev_bits) ||
          mean_bits != mean_bits_ || stddev_bits != stddev_bits_) return 0;
    }
    // Full c10 Stream identity, read from the actual target backend's TLS.
    if (c10::cuda::getCurrentCUDAStream(out.get_device()).unwrap() != stream_) return 0;
    const auto source_elements = outputs_per_unit_ * ((relative - residue_) / stride_);
    {
      py::gil_scoped_release release;
      const auto status = hggcMemcpyAsync(
          out.data_ptr(), source_pointer_ + source_elements * itemsize_, n * itemsize_,
          hggcMemcpyDeviceToDevice, runtime_stream_);
      if (status) {
        throw std::runtime_error("hggcMemcpyAsync returned " + std::to_string(static_cast<int>(status)));
      }
      out.record_stream(stream_);
    }
    return words;
  }

  py::dict metadata() const {
    auto result = stream_metadata(stream_);
    result["raw_start"] = raw_start_;
    result["raw_end"] = raw_end_;
    result["stride"] = stride_;
    result["residue"] = residue_;
    result["outputs_per_unit"] = outputs_per_unit_;
    result["source_pointer"] = reinterpret_cast<std::uintptr_t>(source_pointer_);
    result["source_numel"] = source_.numel();
    result["mean_bits"] = mean_bits_;
    result["stddev_bits"] = stddev_bits_;
    return result;
  }

 private:
  py::object raw_object_;
  at::Tensor raw_;
  at::Tensor source_;
  py::object entry_;
  py::object stream_object_;
  c10::Stream stream_;
  py::object seed_;
  py::object mode_;
  std::int64_t raw_start_, raw_end_, stride_, residue_, outputs_per_unit_;
  std::uint64_t mean_bits_, stddev_bits_;
  at::ScalarType dtype_;
  bool pair_, f64_;
  const char* source_pointer_;
  std::size_t itemsize_;
  hggcStream_t runtime_stream_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("copy_tensor", &copy_tensor, pybind11::call_guard<pybind11::gil_scoped_release>());
  // Diagnostic only; this is never called inside ordinary timing.
  module.def("stream_identity", &stream_identity);
  py::class_<CachedOutput>(module, "CachedOutput")
      .def(py::init<py::object, const at::Tensor&, py::object, py::object, py::object,
                    std::int64_t, py::object, std::int64_t, std::int64_t, std::int64_t,
                    std::uint64_t, std::uint64_t>())
      .def("try_copy", &CachedOutput::try_copy)
      .def("metadata", &CachedOutput::metadata);
  module.def("current_stream_identity", [](c10::DeviceIndex device) {
    return stream_metadata(c10::cuda::getCurrentCUDAStream(device).unwrap());
  });
  module.def("parameter_bits", [](py::handle value, bool f64) -> py::object {
    std::uint64_t bits;
    if (!float_bits(value, f64, bits)) return py::none();
    return py::int_(bits);
  });
}
