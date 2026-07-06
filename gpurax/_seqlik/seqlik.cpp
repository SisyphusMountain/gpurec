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

// coraxlib's own fully-validated SPR wrapper. It is exported (C linkage) by
// the linked coraxlib but, unlike corax_utree_spr, is not declared in any
// public header, so we forward-declare it here. Unlike the bare
// corax_utree_spr (which only rejects a tip prune edge and then blindly
// rewires — segfaulting on, e.g., a regraft that lies inside the pruned
// subtree), corax_utree_spr_safe checks every failure scenario (null nodes,
// tip prune, same-tree no-op, regraft-in-pruned-subtree) and returns
// CORAX_FAILURE instead of corrupting/crashing. apply_spr() uses it so that
// any invalid (prune, regraft) pair raises a Python-catchable exception.
extern "C" int corax_utree_spr_safe(corax_unode_t *p, corax_unode_t *r,
                                    corax_tree_rollback_t *rollback_info);

// True identity-move test, exactly as GeneRax's sprYeldsSameTree
// (SPRSearch.cpp:47 / SearchUtils.cpp:6): a move (prune p, regraft r)
// reproduces the SAME tree iff r is p itself, one of p's rotations, or one
// of p's three incident edges' back pointers. This is a per-DIRECTED-EDGE
// (rotation) check on raw pointers — NOT a logical-node-position check. An
// earlier version canonicalized both prune and regraft to their logical
// array position before this test, which over-rejected: on symmetric /
// balanced trees, a genuinely distinct regraft edge whose logical node
// happened to coincide (under the tree's symmetry) with a no-op edge's
// logical node was discarded, so those trees enumerated ZERO neighbors.
// Comparing rotation pointers directly (like the reference) rejects only
// the six true identity rotations and keeps every genuine move.
static bool isNoOpSprMove(corax_unode_t *p, corax_unode_t *r) {
  return (r == p) || (r == p->next) || (r == p->next->next) ||
         (r == p->back) || (r == p->next->back) ||
         (r == p->next->next->back);
}

// Builds a map from every directed subnode's node_index field to its
// corax_unode_t* rotation pointer, over ALL rotations of every node (not
// just the one canonical rotation stored in _tree->nodes[]). node_index is
// unique per rotation and stable for the life of the tree (it is a fixed
// field, untouched by corax_utree_spr rewiring), so this map lets apply_spr
// resolve the node_index values reported by spr_neighbors back to the exact
// directed edge to prune/regraft. This mirrors GeneRax's
// JointTree::getNode(i) == treeinfo->subnodes[i] (subnodes[i]->node_index
// == i), which is the index space SPRMove records — as opposed to
// PLLUnrootedTree::getNode(i) == _tree->nodes[i], which only reaches one
// rotation per logical node.
static std::unordered_map<unsigned, corax_unode_t *>
buildSubnodeMap(PLLUnrootedTree &tree) {
  std::unordered_map<unsigned, corax_unode_t *> m;
  for (auto *n : tree.getNodes()) {
    m[n->node_index] = n;
    if (n->next) {
      m[n->next->node_index] = n->next;
      m[n->next->next->node_index] = n->next->next;
    }
  }
  return m;
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
  // candidate prune edge. Mirrors GeneRax's enumeration exactly:
  // getAllPruneIndices (SPRSearch.cpp) uses EVERY inner-node rotation as a
  // distinct prune candidate (pruning at rotation p vs p->next vs
  // p->next->next removes three DIFFERENT subtrees), and
  // diggBestMoveFromPrune / diggRecursive (SearchUtils.cpp) seed the regraft
  // walk from prune->next->back and prune->next->next->back, then recurse
  // over regraft->next->back / regraft->next->next->back, pushing an index
  // onto the path at each hop. We only record moves — no scoring, no tree
  // mutation.
  //
  // Every index (prune, regraft, and path entries) is a per-DIRECTED-EDGE
  // `corax_unode_t::node_index`, the SAME index space GeneRax's SPRMove
  // records and that apply_spr() resolves via buildSubnodeMap(). An earlier
  // version collapsed both prune and regraft to one canonical rotation per
  // logical node (PLLUnrootedTree::getNode array positions), which discarded
  // two thirds of all prune and regraft targets and, combined with a
  // logical-position no-op test, enumerated ZERO neighbors on balanced /
  // symmetric trees. Working at rotation granularity fixes that.
  //
  // As in GeneRax (SPRSearch.cpp:304 passes radius + 1 to
  // diggBestMoveFromPrune), we use maxRadius = radius + 1: the seed regraft
  // (prune->next->back) is always a trivial no-op, so genuine moves first
  // appear one hop further out.
  std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>>
  spr_neighbors(unsigned radius) {
    std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>> out;
    auto &tree = _eval.getGeneTree();
    unsigned maxRadius = radius + 1;

    // Every inner-node rotation is a prune candidate (getAllPruneIndices).
    // getNodes() yields one canonical pointer per logical node in stable
    // array order; expand each inner node to its three rotations. Collect
    // and sort by node_index so the prune order is deterministic
    // independent of pointer addresses.
    std::vector<corax_unode_t *> prunes;
    for (auto *n : tree.getNodes()) {
      if (n->next) {  // inner node: three rotations are three prune edges
        prunes.push_back(n);
        prunes.push_back(n->next);
        prunes.push_back(n->next->next);
      }
    }
    std::sort(prunes.begin(), prunes.end(),
              [](corax_unode_t *a, corax_unode_t *b) {
                return a->node_index < b->node_index;
              });

    std::vector<unsigned> path;
    for (auto *prune : prunes) {
      collectSprNeighbors(tree, prune, prune->next->back, path, 0, maxRadius,
                          out);
      collectSprNeighbors(tree, prune, prune->next->next->back, path, 0,
                          maxRadius, out);
    }

    // Verification pass: drop any structurally-enumerated candidate whose
    // applied topology equals the original. The structural sprYeldsSameTree
    // filter above only catches the six trivial adjacent-edge no-ops; two
    // further cases reproduce the original UNROOTED topology hash and must
    // not be reported as neighbors:
    //   (a) automorphism moves on a symmetric tree (regrafting into a
    //       position that maps the tree onto an isomorphic copy of itself);
    //   (b) collisions of GeneRax's getUnrootedTreeHash (its m*i + M mixing
    //       is not injective — distinct labeled topologies can share a hash).
    // Both make apply_spr yield tree_hash() == original, so applying,
    // hashing, and rolling back each candidate is the shape-agnostic
    // identity test that keeps the returned set consistent with a
    // tree_hash-based ground truth. A LOCAL rollback record is used so the
    // public apply_spr()/rollback() state (_lastRollback/_hasMove) is
    // untouched.
    size_t origHash = tree.getUnrootedTreeHash();
    auto subnodes = buildSubnodeMap(tree);
    std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>> verified;
    verified.reserve(out.size());
    for (auto &mv : out) {
      corax_unode_t *p = subnodes.at(std::get<0>(mv));
      corax_unode_t *r = subnodes.at(std::get<1>(mv));
      corax_tree_rollback_t rb{};
      if (corax_utree_spr_safe(p, r, &rb) != CORAX_SUCCESS) {
        continue;  // rejected by coraxlib as invalid/no-op
      }
      bool changed = tree.getUnrootedTreeHash() != origHash;
      corax_tree_rollback(&rb);
      if (changed) {
        verified.push_back(mv);
      }
    }
    // The traversal order is already deterministic, but sort by
    // (prune, regraft, path) as a belt-and-braces stable ordering.
    std::sort(verified.begin(), verified.end());
    return verified;
  }

  // Applies the SPR move (prune_index, regraft_index) in place, recording
  // rollback info. prune_index/regraft_index are per-rotation node_index
  // values, as returned by spr_neighbors().
  //
  // Bounds-checked: an index with no corresponding directed subnode (e.g. a
  // stray/garbage Python-side value) is rejected via buildSubnodeMap() before
  // any dereference, raising a Python-catchable exception instead of indexing
  // out of bounds (previously SIGSEGV). Uses corax_utree_spr_safe, which
  // additionally validates the move itself (no-op moves and regrafts inside
  // the pruned subtree are rejected as CORAX_FAILURE rather than corrupting
  // the tree / crashing), so every invalid pair raises cleanly.
  void apply_spr(unsigned prune_index, unsigned regraft_index) {
    auto &tree = _eval.getGeneTree();
    auto subnodes = buildSubnodeMap(tree);
    auto pit = subnodes.find(prune_index);
    auto rit = subnodes.find(regraft_index);
    if (pit == subnodes.end() || rit == subnodes.end()) {
      throw std::runtime_error("apply_spr: node index out of range");
    }
    int rc = corax_utree_spr_safe(pit->second, rit->second, &_lastRollback);
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

  // Recursive helper for spr_neighbors, mirroring SearchUtils.cpp:
  // diggRecursive. `prune` and `regraft` are directed subnode pointers.
  // Records the move (by node_index) if it is within radius and not a
  // no-op, then, if the radius budget allows and `regraft` is an inner
  // node, recurses one hop further out via regraft->next->back and
  // regraft->next->next->back (never back toward the prune, so the walk
  // stays on the kept side and never enters the pruned subtree).
  static void collectSprNeighbors(
      PLLUnrootedTree &tree, corax_unode_t *prune, corax_unode_t *regraft,
      std::vector<unsigned> &path, unsigned radius, unsigned maxRadius,
      std::vector<std::tuple<unsigned, unsigned, std::vector<unsigned>>> &out) {
    if (radius >= maxRadius) {
      return;
    }
    if (!isNoOpSprMove(prune, regraft)) {
      out.emplace_back(prune->node_index, regraft->node_index, path);
    }
    radius += 1;
    if (regraft->next && radius < maxRadius) {
      path.push_back(regraft->node_index);
      collectSprNeighbors(tree, prune, regraft->next->back, path, radius,
                          maxRadius, out);
      collectSprNeighbors(tree, prune, regraft->next->next->back, path,
                          radius, maxRadius, out);
      path.pop_back();
    }
  }
};

PYBIND11_MODULE(_impl, m) {
  m.doc() = "gpurax sequence-likelihood extension (coraxlib + GeneRaxCore)";
  m.def("corax_version", &corax_version,
        "Return the coraxlib version string this extension was linked against.");

  // GeneRax "Step 0": builds a starting gene tree (random topology, taxa
  // drawn from the alignment) via LibpllEvaluation::createAndSaveRandomTree
  // and writes it to `out`. Used when a family has no starting_tree of its
  // own; the joint SPR search subsequently improves this random start.
  m.def("build_starting_tree",
        [](const std::string &aln, const std::string &model,
           const std::string &out) {
          LibpllEvaluation::createAndSaveRandomTree(aln, model, out);
          return out;
        },
        py::arg("alignment"), py::arg("model"), py::arg("out"));

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
