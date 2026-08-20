#include "flagrand/utils.h"

#include <dlfcn.h>
#include <stdexcept>

namespace flagrand::utils {

std::filesystem::path get_path_of_operators_library() {
  static const std::filesystem::path path = [] {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&get_path_of_operators_library), &info) && info.dli_fname) {
      return std::filesystem::canonical(info.dli_fname);
    }
    throw std::runtime_error("cannot resolve the FlagRAND operators library path");
  }();
  return path;
}

std::filesystem::path get_triton_source_path() {
  return get_path_of_operators_library().parent_path() / "triton_src" / "philox.py";
}

}  // namespace flagrand::utils
