#include <pybind11/pybind11.h>
#include "torch/python.h"

#include "flagrand/operators.h"

namespace py = pybind11;

PYBIND11_MODULE(c_operators, module) {
  module.doc() = "FlagRAND C++ Philox Host API backed by TritonJIT";

  py::class_<flagrand::Generator, flagrand::GeneratorPtr>(module, "Generator")
      .def_property_readonly("seed", &flagrand::Generator::seed)
      .def_property_readonly("offset", &flagrand::Generator::offset)
      .def("set_seed", &flagrand::Generator::set_seed)
      .def("set_offset", &flagrand::Generator::set_offset);

  module.def("create_generator", &flagrand::create_generator, py::arg("seed") = 0, py::arg("offset") = 0);
  module.def("generate", [](const flagrand::GeneratorPtr& gen, const at::Tensor& out) {
    return flagrand::generate(*gen, out);
  });
  module.def("generate_uniform", [](const flagrand::GeneratorPtr& gen, const at::Tensor& out) {
    return flagrand::generate_uniform(*gen, out);
  });
  module.def("generate_uniform_double", [](const flagrand::GeneratorPtr& gen, const at::Tensor& out) {
    return flagrand::generate_uniform_double(*gen, out);
  });
  module.def(
      "generate_normal",
      [](const flagrand::GeneratorPtr& gen, const at::Tensor& out, double mean, double stddev) {
        return flagrand::generate_normal(*gen, out, mean, stddev);
      },
      py::arg("generator"), py::arg("out"), py::arg("mean"), py::arg("stddev"));
  module.def(
      "generate_lognormal",
      [](const flagrand::GeneratorPtr& gen, const at::Tensor& out, double mean, double stddev) {
        return flagrand::generate_lognormal(*gen, out, mean, stddev);
      },
      py::arg("generator"), py::arg("out"), py::arg("mean"), py::arg("stddev"));
}
