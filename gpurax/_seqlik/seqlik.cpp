// Minimal pybind11 module for the gpurax sequence-likelihood extension.
//
// Task A2 scope: prove that coraxlib + the GeneRaxCore sources it needs
// link cleanly into a pybind11 module. Only `corax_version()` is exposed;
// later tasks (B1+) add the real bindings (SeqFamily, etc.) to this file.
//
// Task B1: SeqFamily wraps LibpllEvaluation to load a gene family (newick +
// alignment + substitution model) and compute its sequence log-likelihood
// (natural-log units, i.e. libpll's native units == GeneRax's reported
// units — no bits conversion here).

#include <pybind11/pybind11.h>

#include <corax/corax.h>

#include <string>

#include "likelihoods/LibpllEvaluation.hpp"
#include "trees/PLLUnrootedTree.hpp"

namespace py = pybind11;

// coraxlib does not expose a version macro/symbol of its own; this string
// simply confirms the extension built and linked against corax + the
// GeneRaxCore sources listed in CMakeLists.txt.
static std::string corax_version() {
  return std::string("corax-linked");
}

// Loads a single gene family (newick string + FASTA alignment + substitution
// model token, e.g. "GTR") and exposes its libpll sequence log-likelihood.
class SeqFamily {
public:
  SeqFamily(const std::string &newick, const std::string &alignment,
            const std::string &model)
      : _eval(newick, /*isNewickAFile=*/false, alignment, model) {}

  // LL at current branch lengths / model params (natural log). If opt_bl is
  // true, branch lengths (only) are optimized first.
  double seq_loglk(bool opt_bl) {
    if (opt_bl) {
      _eval.optimizeBranches();
    }
    return _eval.computeLikelihood(/*incremental=*/false);
  }

  // Optimizes both substitution-model parameters and branch lengths, then
  // returns the resulting log-likelihood (natural log). This is the
  // apples-to-apples comparison with a GeneRax EVAL run, which optimizes
  // both before reporting its libpll likelihood.
  double optimize_all() { return _eval.optimizeAllParameters(); }

  std::string newick() { return _eval.getGeneTree().getNewickString(); }

private:
  LibpllEvaluation _eval;
};

PYBIND11_MODULE(_impl, m) {
  m.doc() = "gpurax sequence-likelihood extension (coraxlib + GeneRaxCore)";
  m.def("corax_version", &corax_version,
        "Return the coraxlib version string this extension was linked against.");

  py::class_<SeqFamily>(m, "SeqFamily")
      .def(py::init<const std::string &, const std::string &, const std::string &>(),
           py::arg("newick"), py::arg("alignment_path"), py::arg("model"))
      .def("seq_loglk", &SeqFamily::seq_loglk, py::arg("opt_bl") = false)
      .def("optimize_all", &SeqFamily::optimize_all)
      .def("newick", &SeqFamily::newick);
}
