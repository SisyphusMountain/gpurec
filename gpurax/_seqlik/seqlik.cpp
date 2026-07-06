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
#include <pybind11/stl.h>

#include <corax/corax.h>
#include <corax/tree/utree_moves.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "likelihoods/LibpllEvaluation.hpp"
#include "trees/PLLUnrootedTree.hpp"

namespace py = pybind11;

// Rotation-invariant replacement for the pointer-identity check copied from
// SPRSearch.cpp:47 (identical in SearchUtils.cpp). The original check
// compared raw pointers: (r == p) || (r == p->next) || (r == p->next->next)
// || (r == p->back) || (r == p->next->back) || (r == p->next->next->back).
// That is only correct when `r` is literally the same rotation object
// reached by walking from `p`. spr_neighbors() canonicalizes both prune and
// regraft to their array-stored (canonical) rotation via positionOfField
// before this check runs, so a genuinely-adjacent-but-differently-rotated
// neighbor no longer pointer-matches p->back / p->next->back /
// p->next->next->back, and identity (no-op) moves slip through undetected.
// Fix: compare LOGICAL identity, i.e. canonical array positions, not
// pointers. `prunePos`/`regraftPos` are already canonical array positions
// (as built by spr_neighbors' positionOfField map); this recomputes the
// canonical positions of prune's own node and its three neighbors and
// compares against regraftPos.
static bool isNoOpSprMove(
    PLLUnrootedTree &tree,
    const std::unordered_map<unsigned, unsigned> &positionOfField,
    unsigned prunePos, unsigned regraftPos) {
  if (regraftPos == prunePos) {
    return true;  // regraft target is prune's own logical node
  }
  corax_unode_t *prune = tree.getNode(prunePos);
  unsigned n0 = positionOfField.at(prune->back->node_index);
  unsigned n1 = positionOfField.at(prune->next->back->node_index);
  unsigned n2 = positionOfField.at(prune->next->next->back->node_index);
  return regraftPos == n0 || regraftPos == n1 || regraftPos == n2;
}

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

  // Enumerates SPR neighbor moves up to `radius` regraft hops away from each
  // candidate prune edge. Mirrors the recursion pattern in
  // SearchUtils.cpp:diggRecursive (regraft over next->back / next->next->back,
  // pushing an index onto the path at each hop) but only records moves — no
  // scoring, no tree mutation. Returns a flat list of
  // (prune_index, regraft_index, path) tuples, where every index is an
  // ARRAY POSITION suitable for PLLUnrootedTree::getNode() (see below).
  //
  // Two corrections relative to a literal transcription of diggRecursive:
  //  1. The seed regraft node passed into the recursion (prune->next->back /
  //     prune->next->next->back) is *always* one of isNoOpSprMove's own
  //     checks, so it is trivially rejected as a no-op; real candidates only
  //     appear one hop further out. SPRSearch::applySPRRound (SPRSearch.cpp:
  //     304) accounts for this by passing `radius + 1` as maxRadius to
  //     diggBestMoveFromPrune — we do the same for our user-facing `radius`.
  //  2. PLLUnrootedTree::getNode(i) does plain array indexing
  //     (`_tree->nodes[i]`), but a node's own `corax_unode_t::node_index`
  //     field is NOT reliably equal to its position in that array: verified
  //     empirically that, past the first internal node, node_index can lag
  //     the true array position (e.g. a 6-taxon tree here has internal nodes
  //     at positions 7/8/9 whose own node_index fields read 9/12/15).
  //     Recording those raw field values and later feeding them back into
  //     getNode() (as apply_spr does) then indexes out of bounds. The field
  //     *is* shared across an internal node's (up to 3) rotations, though, so
  //     we build a one-time field->position map by scanning tree.getNodes()
  //     (which yields exactly one canonical pointer per logical node, at its
  //     true array position) and use that to canonicalize/convert every node
  //     we touch during traversal to its true array position. Prune/regraft
  //     indices exposed to Python — and expected back by apply_spr() — are
  //     therefore always array positions, not raw node_index fields.
  std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>>
  spr_neighbors(unsigned radius) {
    std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>> out;
    auto &tree = _eval.getGeneTree();
    unsigned maxRadius = radius + 1;

    // Maps EVERY rotation's node_index field to its logical node's true
    // array position, not just the canonical (array-stored) rotation's own
    // field. This matters because a node's up to 3 rotations do NOT share
    // one common node_index value (verified empirically: e.g. an internal
    // node whose canonical rotation is at array position 6 has rotations
    // with node_index 6/7/8 — three distinct values, not one shared value).
    // Any pointer walk (->back, ->next, ...) can land on a non-canonical
    // rotation, so every rotation's field must resolve to the same
    // position for lookups like isNoOpSprMove()'s prune->back/next->back
    // checks to work regardless of which rotation was reached.
    std::unordered_map<unsigned, unsigned> positionOfField;
    unsigned pos = 0;
    for (auto *n : tree.getNodes()) {
      positionOfField[n->node_index] = pos;
      if (n->next) {
        positionOfField[n->next->node_index] = pos;
        positionOfField[n->next->next->node_index] = pos;
      }
      pos++;
    }

    std::unordered_set<unsigned> seenPrunePositions;
    for (auto *rawPrune : tree.getBranches()) {
      unsigned prunePos = positionOfField.at(rawPrune->node_index);
      if (!seenPrunePositions.insert(prunePos).second) {
        continue;  // already tried this logical node as a prune candidate
      }
      corax_unode_t *prune = tree.getNode(prunePos);
      // corax_utree_spr requires the prune edge to be defined by an inner
      // node (getAllPruneIndices in SPRSearch.cpp applies the same filter).
      if (!prune->next) {
        continue;
      }
      std::vector<unsigned> path;
      collectSprNeighbors(tree, positionOfField, prune->next->back,
                          path, 1, maxRadius, prunePos, out);
      collectSprNeighbors(tree, positionOfField,
                          prune->next->next->back, path, 1, maxRadius,
                          prunePos, out);
    }
    // getBranches() is backed by an unordered_set keyed on pointer address,
    // so iteration order (and therefore the order moves are discovered in)
    // is nondeterministic across runs/processes. Sort by (prune, regraft,
    // path) before returning so the result is stable.
    std::sort(out.begin(), out.end());
    return out;
  }

  // Applies the SPR move (prune_index, regraft_index) in place, recording
  // rollback info. Throws on failure instead of silently corrupting the
  // tree (e.g. invalid/tip prune node, or a no-op move rejected by
  // coraxlib). prune_index/regraft_index are array positions, as returned
  // by spr_neighbors().
  //
  // Bounds-checked: getNode() does raw `nodes[idx]` indexing with no bounds
  // checking of its own, so an out-of-range index (e.g. a stray/garbage
  // Python-side value) previously dereferenced past the end of the node
  // array (SIGSEGV). Validate both indices against the tree's actual node
  // count first and raise a Python-catchable exception instead.
  void apply_spr(unsigned prune_index, unsigned regraft_index) {
    auto &tree = _eval.getGeneTree();
    unsigned nNodes = tree.getNodeNumber();
    if (prune_index >= nNodes || regraft_index >= nNodes) {
      throw std::runtime_error("apply_spr: node index out of range");
    }
    corax_unode_t *pe = tree.getNode(prune_index);
    corax_unode_t *re = tree.getNode(regraft_index);
    int rc = corax_utree_spr(pe, re, &_lastRollback);
    if (rc != CORAX_SUCCESS) {
      throw std::runtime_error("corax_utree_spr failed");
    }
    _hasMove = true;
  }

  // Undoes the most recent apply_spr() call. Throws if there is no pending
  // move to undo, instead of running corax_tree_rollback() on a
  // (previously uninitialized) rollback record — that read UB garbage and
  // switched on a garbage rearrange_type.
  void rollback() {
    if (!_hasMove) {
      throw std::runtime_error("rollback: no move to undo");
    }
    corax_tree_rollback(&_lastRollback);
    _hasMove = false;
  }

  // Hash of the current unrooted topology (branch lengths do not affect it).
  size_t tree_hash() { return _eval.getGeneTree().getUnrootedTreeHash(); }

private:
  LibpllEvaluation _eval;
  // Zero-initialized: previously left uninitialized, so a rollback() call
  // before any apply_spr() ran corax_tree_rollback() on garbage (UB). Now
  // guarded additionally by _hasMove (see rollback()/apply_spr() above).
  corax_tree_rollback_t _lastRollback{};
  bool _hasMove = false;

  // Recursive helper for spr_neighbors: canonicalizes `regraftRaw` to its
  // true array position (see the note on spr_neighbors above), and if
  // (prune, regraft) is not a no-op move, records it; then, if the radius
  // budget allows and `regraft` is an inner node, recurses one hop further
  // via regraft->next->back and regraft->next->next->back.
  static void collectSprNeighbors(
      PLLUnrootedTree &tree,
      const std::unordered_map<unsigned, unsigned> &positionOfField,
      corax_unode_t *regraftRaw,
      std::vector<unsigned> &path, unsigned radius, unsigned maxRadius,
      unsigned prunePos,
      std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>> &out) {
    unsigned regraftPos = positionOfField.at(regraftRaw->node_index);
    corax_unode_t *regraft = tree.getNode(regraftPos);
    if (!isNoOpSprMove(tree, positionOfField, prunePos, regraftPos)) {
      out.emplace_back(prunePos, regraftPos, path);
    }
    if (radius < maxRadius && regraft->next) {
      path.push_back(regraftPos);
      collectSprNeighbors(tree, positionOfField, regraft->next->back,
                          path, radius + 1, maxRadius, prunePos, out);
      collectSprNeighbors(tree, positionOfField,
                          regraft->next->next->back, path, radius + 1,
                          maxRadius, prunePos, out);
      path.pop_back();
    }
  }
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
      .def("newick", &SeqFamily::newick)
      .def("spr_neighbors", &SeqFamily::spr_neighbors, py::arg("radius"))
      .def("apply_spr", &SeqFamily::apply_spr, py::arg("prune_index"),
           py::arg("regraft_index"))
      .def("rollback", &SeqFamily::rollback)
      .def("tree_hash", &SeqFamily::tree_hash);
}
