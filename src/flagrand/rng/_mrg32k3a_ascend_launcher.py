"""Private prepared C launches for Ascend MRG32K3A FP32 kernels.

The installed NPULauncher still owns stream, hook, registration and profiler
handling. Only its private generated C entry and queued handler are adapted.
"""

from pathlib import Path
from types import FunctionType

POINTER_HELPER = r"""
// FLAGRAND_MRG32_ASCEND_NATIVE_ENTRY
#include <torch/csrc/autograd/python_variable.h>
#include <c10/core/DeviceType.h>
#include <limits>

static inline DevicePtrInfo getPointerPrepared(PyObject *obj, int idx) {
  if (!THPVariable_CheckExact(obj)) return getPointer(obj, idx);
  DevicePtrInfo info;
  const at::Tensor& tensor = THPVariable_Unpack(obj);
  info.dev_ptr = tensor.data_ptr();
  info.valid = true;
  if (!info.dev_ptr) return info;
  if (tensor.device().type() == c10::DeviceType::PrivateUse1) return info;
  aclrtPtrAttributes attributes;
  aclError status = aclrtPointerGetAttributes(info.dev_ptr, &attributes);
  if (status == ACL_SUCCESS) {
    if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE && attributes.location.type != 4) {
      PyErr_Format(PyExc_ValueError,
                   "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      info.valid = false;
    }
  } else {
    PyErr_Format(PyExc_RuntimeError,
                 "Failed to query pointer attributes at argument %d. Error code: %d. "
                 "This may indicate invalid memory address or NPU device error.", idx, status);
    info.valid = false;
  }
  return info;
}

static inline int prepared_i32(PyObject *value) {
  long result = PyLong_AsLong(value);
  if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max()) {
    PyErr_SetString(PyExc_OverflowError, "signed integer is out of range");
    return -1;
  }
  return static_cast<int>(result);
}
"""

FAST_PREFIX = r"""
static PyObject* launch_prepared(PyObject* self, PyObject* const* values, Py_ssize_t nargs) {
  HANDLE_TH_ERRORS
  if (nargs != 17) {
    PyErr_Format(PyExc_TypeError, "expected 17 launch arguments, got %zd", nargs);
    return NULL;
  }
  int gridX = prepared_i32(values[0]);
  int gridY = prepared_i32(values[1]);
  int gridZ = prepared_i32(values[2]);
  rtStream_t stream = reinterpret_cast<rtStream_t>(PyLong_AsUnsignedLongLongMask(values[3]));
  const void *function = reinterpret_cast<const void*>(PyLong_AsUnsignedLongLongMask(values[4]));
  PyObject *packedMetadata = values[5];
  PyObject *launch_metadata = values[6];
  PyObject *launch_enter_hook = values[7];
  PyObject *launch_exit_hook = values[8];
  std::vector<std::vector<int64_t>> tensorShapes;
  PyObject *_arg0 = values[9];
  PyObject *_arg1 = values[10];
  PyObject *_arg2 = values[11];
  int32_t _arg3 = prepared_i32(values[12]);
  int32_t _arg4 = prepared_i32(values[13]);
  int32_t _arg5 = prepared_i32(values[14]);
  float _arg6 = static_cast<float>(PyFloat_AsDouble(values[15]));
  float _arg7 = static_cast<float>(PyFloat_AsDouble(values[16]));
  if (PyErr_Occurred()) return NULL;
"""

QUEUE_BRIDGE = r"""
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cstdint>

extern uint32_t OpApiGetTaskQueueEnable();

static void launch_on_current_queue(void *func_obj, const char *name) {
  auto current = c10_npu::getCurrentNPUStream();
  if (current.isSyncLaunchStream() || !OpApiGetTaskQueueEnable()) {
    g_async_launch(func_obj, name);
    return;
  }
  // V1 owns the copied handler and retains the ordinary task queue, profiler
  // marks and error path. Compiled RNG does not request generic aclop/HCCL
  // deterministic-setting side effects.
  auto func = *static_cast<std::function<rtError_t()> *>(func_obj);
  at_npu::native::OpCommand::RunOpApi(name, func, false);
}

"""


def _prepare_source(source):
    begin = source.index("static PyObject* launch(PyObject* self, PyObject* args)")
    end = source.index("static PyMethodDef ModuleMethods", begin)
    ordinary = source[begin:end]
    tail = ordinary[ordinary.index("  if (__MsprofFlagL1)") :]
    tail = tail[: tail.rfind("}")].replace("getPointer(", "getPointerPrepared(")
    prepared = POINTER_HELPER + FAST_PREFIX + tail + "  END_HANDLE_TH_ERRORS\n}\n\n"
    result = source[:end] + prepared + source[end:]
    method = '{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}'
    result = result.replace(
        method,
        method + ',\n  {"launch_prepared", reinterpret_cast<PyCFunction>(launch_prepared), METH_FASTCALL, '
        '"Prepared real-Tensor entry"}',
        1,
    )
    if "std::function<rtError_t()> launch_call =" in result:
        start = result.index("static void _launch(")
        prefix, suffix = result[:start], result[start:]
        suffix = suffix.replace(
            "g_async_launch(static_cast<void*>(&launch_call), name.c_str());",
            "launch_on_current_queue(static_cast<void*>(&launch_call), name.c_str());",
            1,
        )
        result = prefix + QUEUE_BRIDGE + suffix
    return result


def _eligible(constants, signature, metadata):
    distribution = constants.get(10)
    if distribution in (0, 1):
        output = signature.get(1)
        spare = "*i32"
        output_ok = output in ("*i32", "*fp32")
    elif distribution in (2, 3):
        output, spare, output_ok = "*fp32", "*fp32", True
    else:
        return False
    expected = {0: "*i32", 1: output, 2: spare, 3: "i32", 4: "i32", 5: "i32", 6: "fp32", 7: "fp32"}
    return metadata.name == "generate32_kernel" and output_ok and signature == expected


def _clone_function(function, globals):
    copied = FunctionType(function.__code__, globals, function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    copied.__annotations__ = dict(function.__annotations__)
    copied.__qualname__ = function.__qualname__
    return copied


class _EntryFactory:
    def __init__(self):
        import torch
        import torch_npu
        from triton.runtime.driver import driver

        base = driver.active.launcher_cls
        method = base._make_launcher_stub_path
        owner = method.__globals__
        generate = owner["generate_npu_wrapper_src"]
        stub = owner["make_npu_launcher_stub"]
        build = stub.__globals__["_build_npu_ext"]
        backend = build.__globals__["get_backend_func"]
        torch_root = Path(torch.__file__).resolve().parent
        npu_root = Path(torch_npu.__file__).resolve().parent

        def private_backend(name, *args, **kwargs):
            result = backend(name, *args, **kwargs)
            if name == "get_cc_cmd":
                result = (
                    list(result)
                    + list(backend("get_cc_cmd_npu_utils"))
                    + [
                        f"-I{torch_root / 'include'}",
                        f"-I{torch_root / 'include/torch/csrc/api/include'}",
                        f"-L{torch_root / 'lib'}",
                        f"-Wl,-rpath,{torch_root / 'lib'}",
                        f"-Wl,-rpath,{npu_root / 'lib'}",
                        "-ltorch_python",
                        "-ltorch",
                        "-ltorch_cpu",
                        "-lc10",
                    ]
                )
            return result

        private_build = _clone_function(build, dict(build.__globals__, get_backend_func=private_backend))
        private_stub = _clone_function(stub, dict(stub.__globals__, _build_npu_ext=private_build))

        def private_generate(constants, signature, metadata):
            return _prepare_source(generate(constants, signature, metadata))

        private_method = _clone_function(
            method,
            dict(owner, generate_npu_wrapper_src=private_generate, make_npu_launcher_stub=private_stub),
        )

        class PrivateNPULauncher(base):
            _make_launcher_stub_path = private_method

            def __init__(self, src, metadata):
                super().__init__(src, metadata)
                self.launch = self.launch.__self__.launch_prepared

        self.launcher_class = PrivateNPULauncher

    def __call__(self, compiled, original_run):
        src = compiled.src
        index = lambda key: src.fn.arg_names.index(key) if isinstance(key, str) else key
        constants = {index(key): value for key, value in src.constants.items()}
        signature = {index(key): value for key, value in src.signature.items()}
        if not _eligible(constants, signature, compiled.metadata):
            return original_run
        private = self.launcher_class(src, compiled.metadata)
        private.compile_only = original_run.compile_only
        private.enable_msprof_register_tensor = original_run.enable_msprof_register_tensor
        return private


def create_entry_factory():
    return _EntryFactory()
