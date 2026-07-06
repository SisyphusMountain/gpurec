// Minimal pybind11 module for the gpurax sequence-likelihood extension.
//
// Task A2 scope: prove that coraxlib + the GeneRaxCore sources it needs
// link cleanly into a pybind11 module. Only `corax_version()` is exposed;
// later tasks (B1+) add the real bindings (SeqFamily, etc.) to this file.

#include <pybind11/pybind11.h>

#include <corax/corax.h>

#include <string>

namespace py = pybind11;

// coraxlib does not expose a version macro/symbol of its own; this string
// simply confirms the extension built and linked against corax + the
// GeneRaxCore sources listed in CMakeLists.txt.
static std::string corax_version() {
  return std::string("corax-linked");
}

PYBIND11_MODULE(_impl, m) {
  m.doc() = "gpurax sequence-likelihood extension (coraxlib + GeneRaxCore)";
  m.def("corax_version", &corax_version,
        "Return the coraxlib version string this extension was linked against.");
}
