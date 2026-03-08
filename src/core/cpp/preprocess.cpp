#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tree_utils.hpp"
#include "clade_utils.hpp"

namespace py = pybind11;

namespace {

// ============================================================================
// Constants and Utilities
// ============================================================================

constexpr size_t BITS_PER_WORD = 64;

inline size_t bitvec_num_words(int num_leaves) {
  return (static_cast<size_t>(num_leaves) + BITS_PER_WORD - 1) / BITS_PER_WORD;
}

// Default way of parsing names: assume species name is prefix before first '_'
// TODO: replicate ALERax behavior instead of this.
std::string extract_species_name(const std::string &leaf_name) {
  auto pos = leaf_name.find('_');
  if (pos != std::string::npos) {
    return leaf_name.substr(0, pos);
  }
  return leaf_name;
}

// ============================================================================
// Tensor Conversion Helpers
// ============================================================================

torch::Tensor to_long_tensor(const std::vector<int64_t>& vec) {
  return torch::from_blob(
      const_cast<int64_t*>(vec.data()),
      {static_cast<long>(vec.size())},
      torch::TensorOptions().dtype(torch::kInt64))
      .clone();
}

torch::Tensor to_double_tensor(const std::vector<double>& vec) {
  return torch::from_blob(
      const_cast<double*>(vec.data()),
      {static_cast<long>(vec.size())},
      torch::TensorOptions().dtype(torch::kFloat64))
      .clone();
}

torch::Tensor to_double_matrix(const std::vector<double>& mat, int rows, int cols) {
  return torch::from_blob(
      const_cast<double*>(mat.data()),
      {rows, cols},
      torch::TensorOptions().dtype(torch::kFloat64))
      .clone();
}

torch::Tensor to_uint8_tensor(const std::vector<uint8_t>& vec) {
  return torch::from_blob(
      const_cast<uint8_t*>(vec.data()),
      {static_cast<long>(vec.size())},
      torch::TensorOptions().dtype(torch::kUInt8))
      .clone();
}

// ============================================================================
// Clade and CladeSplit Classes
// ============================================================================

/**
 * @brief Represents a clade (subset of leaves in a tree)
 *
 * A clade is uniquely identified by its BitVec representation.
 * Provides hashing and canonical ordering for deduplication.
 */
class Clade {
private:
  BitVec bits_;
  int size_;

public:
  explicit Clade(const BitVec& bits)
      : bits_(bits), size_(bit_count(bits)) {}
  
  explicit Clade(BitVec&& bits)
      : bits_(std::move(bits)), size_(bit_count(bits_)) {}

  const BitVec& bits() const { return bits_; }
  int size() const { return size_; }
  /**
   * @brief Equality comparison based on BitVec content
   */
  bool operator==(const Clade& other) const {
    return bits_ == other.bits_;
  }

  /**
   * @brief Lexicographic ordering for canonical split keys
   */
  bool operator<(const Clade& other) const {
    if (size_ != other.size_) {
      return size_ < other.size_;
    }
    return bitvec_lex_less(bits_, other.bits_);
  }
};

/**
 * @brief Manages a collection of clades with ID assignment
 */
class CladeRegistry {
private:
  std::vector<Clade> clades_;
  std::unordered_map<BitVec, int, BitVecHash, BitVecEqual> bitvec_to_id_;

public:
  CladeRegistry() {
    clades_.reserve(1024);
    bitvec_to_id_.reserve(1024);
  }

  /**
   * @brief Get or create a clade, returning its ID
   */
  int get_or_create(const BitVec& bits) {
    auto it = bitvec_to_id_.find(bits);
    if (it != bitvec_to_id_.end()) {
      return it->second;
    }
    int new_id = static_cast<int>(clades_.size());
    clades_.emplace_back(bits);
    bitvec_to_id_.emplace(clades_.back().bits(), new_id);
    return new_id;
  }

  int get_or_create(BitVec&& bits) {
    auto it = bitvec_to_id_.find(bits);
    if (it != bitvec_to_id_.end()) {
      return it->second;
    }
    int new_id = static_cast<int>(clades_.size());
    clades_.emplace_back(std::move(bits));
    bitvec_to_id_.emplace(clades_.back().bits(), new_id);
    return new_id;
  }

  const Clade& get(int id) const {
    return clades_[id];
  }

  size_t size() const {
    return clades_.size();
  }

  const std::vector<Clade>& all() const {
    return clades_;
  }
};

/**
 * @brief Represents a split (parent clade divided into two child clades)
 */
class CladeSplit {
private:
  int parent_id_;
  int left_id_;
  int right_id_;
  double weight_;

public:
  CladeSplit(int parent, int left, int right, double weight = 1.0)
      : parent_id_(parent), left_id_(left), right_id_(right), weight_(weight) {}

  int parent() const { return parent_id_; }
  int left() const { return left_id_; }
  int right() const { return right_id_; }
  double weight() const { return weight_; }

  void add_weight(double w) { weight_ += w; }

  /**
   * @brief Create canonical key for deduplication
   *
   * Ensures left_id <= right_id for consistent hashing
   */
  std::tuple<int, int, int> canonical_key() const {
    int l = left_id_;
    int r = right_id_;
    if (l > r) {
      std::swap(l, r);
    }
    return std::make_tuple(parent_id_, l, r);
  }

  /**
   * @brief Create canonical PairKey for root splits
   *
   * Orders by clade size, then lexicographically
   */
  static PairKey canonical_root_key(
      int a_id, const Clade& a,
      int b_id, const Clade& b) {
    if (a.size() < b.size()) {
      return PairKey{a_id, b_id};
    }
    if (a.size() > b.size()) {
      return PairKey{b_id, a_id};
    }
    if (a < b) {
      return PairKey{a_id, b_id};
    }
    if (b < a) {
      return PairKey{b_id, a_id};
    }
    return PairKey{std::min(a_id, b_id), std::max(a_id, b_id)};
  }
};

/**
 * @brief Result structure for clade and split computation
 * 
 * Contains the set of clades, splits, and root clade ID.
 */
struct CladeData {
  CladeRegistry clades;
  std::vector<CladeSplit> splits;
  int root_clade_id;
};


struct CCPArrays {
  std::vector<int64_t> split_parents_sorted;
  std::vector<int64_t> split_lefts_sorted;
  std::vector<int64_t> split_rights_sorted;
  std::vector<int64_t> split_leftrights_sorted;
  std::vector<double> log_split_probs_sorted;
  std::vector<int64_t> parents_sorted;
  std::vector<int64_t> seg_counts;
  std::vector<int64_t> ptr;
  std::vector<int64_t> ptr_ge2;
  std::vector<int64_t> split_counts;
  std::vector<int64_t> split_order;
  int num_segs_ge2;
  int num_segs_eq1;
  int num_segs_eq0;
  int stop_reduce_ptr_idx;
  int end_rows_ge2;

  // Clade inclusion DAG: edge (inclusion_children[i], inclusion_parents[i])
  // means inclusion_children[i] ⊆ inclusion_parents[i]
  std::vector<int64_t> inclusion_children;
  std::vector<int64_t> inclusion_parents;
  int64_t ubiquitous_clade_id;  // Root of the inclusion DAG (contains all leaves)
};

// ============================================================================
// Core Algorithms
// ============================================================================

/**
 * @brief Compute all clades and splits from a gene tree
 *
 * For a tree with n leaves, generates exactly 4n-5 clades:
 * - n singleton clades (one per leaf)
 * - 3(n-2) clades from internal nodes (3 per node of degree 3)
 * - 1 root clade (all leaves)
 */
CladeData compute_clades_and_splits(
    TreeNode *gene_root,
    const std::vector<std::string> &leaf_names,
    const std::unordered_map<std::string, int> &leaf_to_index) {

  CladeData result;
  const int num_leaves = static_cast<int>(leaf_names.size());
  const size_t num_words = bitvec_num_words(num_leaves);

  std::vector<TreeNode *> postorder_nodes;
  collect_nodes_postorder(gene_root, postorder_nodes);
  const size_t num_nodes = postorder_nodes.size();

  std::unordered_map<TreeNode *, size_t> node_index;
  node_index.reserve(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    node_index[postorder_nodes[i]] = i;
  }

  std::vector<BitVec> node_clades(num_nodes, BitVec(num_words, 0ULL));
  std::vector<int> node_clade_ids(num_nodes, -1);
  std::vector<int> node_above_ids(num_nodes, -1);

  // First pass: build clades for each node
  // Will create the n leaf clades, n-2 internal clade nodes (corresponding to this fixed root choice), and the ubiquitous clade
  // We will still have to construct clades corresponding to other rootings.
  for (TreeNode *node : postorder_nodes) {
    size_t idx = node_index[node];
    BitVec bits(num_words, 0ULL);
    if (node->children.empty()) {
      int leaf_idx = leaf_to_index.at(node->name);
      set_bit(bits, leaf_idx);
    } else {
      const BitVec &left = node_clades[node_index[node->children[0]]];
      const BitVec &right = node_clades[node_index[node->children[1]]];
      for (size_t w = 0; w < num_words; ++w) {
        bits[w] = left[w] | right[w];
      }
    }
    node_clades[idx] = bits;
    node_clade_ids[idx] = result.clades.get_or_create(std::move(bits));
  }

  const BitVec &root_bits = node_clades[node_index[gene_root]];
  result.root_clade_id = node_clade_ids[node_index[gene_root]];

  result.splits.reserve(num_nodes * 3);
  std::unordered_set<PairKey, PairKeyHash, PairKeyEqual> root_split_keys;
  root_split_keys.reserve(num_nodes * 2);

  // Second pass: compute "above" clades and root splits
  for (TreeNode *node : postorder_nodes) {
    if (node == gene_root) {
      continue;
    }
    size_t idx = node_index[node];
    const BitVec &below_bits = node_clades[idx];
    BitVec above_bits = bit_difference(root_bits, below_bits);
    if (is_empty(above_bits)) {
      continue;
    }

    int below_id = node_clade_ids[idx];
    int above_id = result.clades.get_or_create(std::move(above_bits));
    node_above_ids[idx] = above_id;

    const Clade& below_clade = result.clades.get(below_id);
    const Clade& above_clade = result.clades.get(above_id);

    PairKey key = CladeSplit::canonical_root_key(below_id, below_clade, above_id, above_clade);
    if (root_split_keys.insert(key).second) {
      result.splits.emplace_back(result.root_clade_id, below_id, above_id, 1.0);
    }
  }

  // Third pass: create internal splits
  for (TreeNode *node : postorder_nodes) {
    if (node == gene_root || node->children.size() != 2) {
      continue;
    }
    size_t idx = node_index[node];
    int parent_id = node_clade_ids[idx];
    TreeNode *left_node = node->children[0];
    TreeNode *right_node = node->children[1];
    size_t left_idx = node_index[left_node];
    size_t right_idx = node_index[right_node];
    int left_id = node_clade_ids[left_idx];
    int right_id = node_clade_ids[right_idx];
    result.splits.emplace_back(parent_id, left_id, right_id, 1.0);

    int above_id = node_above_ids[idx];
    if (above_id >= 0) {
      int left_plus_id = node_above_ids[right_idx];
      int right_plus_id = node_above_ids[left_idx];
      result.splits.emplace_back(left_plus_id, left_id, above_id, 1.0);
      result.splits.emplace_back(right_plus_id, right_id, above_id, 1.0);
    }
  }

  return result;
}

/**
 * @brief Convert CladeData to sorted CCP arrays for efficient computation
 */
CCPArrays build_ccp_arrays(const CladeData& clade_data) {
  CCPArrays result;
  const size_t C = clade_data.clades.size();
  const size_t N_splits = clade_data.splits.size();
  const std::vector<CladeSplit>& splits = clade_data.splits;

  std::vector<int64_t> split_parents(N_splits);
  std::vector<int64_t> split_lefts(N_splits);
  std::vector<int64_t> split_rights(N_splits);
  std::vector<double> split_weights(N_splits);
  result.split_counts.resize(C, 0);
  // Creating the vectors of split components
  // As well as the number of ways each clade is split (split_counts)
  for (size_t i = 0; i < N_splits; ++i) {
    split_parents[i] = splits[i].parent();
    split_lefts[i] = splits[i].left();
    split_rights[i] = splits[i].right();
    split_weights[i] = splits[i].weight();
    result.split_counts[splits[i].parent()] += 1;
  }

  std::vector<double> sum_weights(C, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    sum_weights[split_parents[i]] += split_weights[i];
  }

  std::vector<double> split_probs(N_splits, 0.0);
  std::vector<double> log_split_probs(N_splits, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    double denom = sum_weights[split_parents[i]];
    split_probs[i] = denom > 0.0 ? split_weights[i] / denom : 0.0;
    log_split_probs[i] = split_probs[i] > 0.0
                              ? std::log(split_probs[i])
                              : -std::numeric_limits<double>::infinity();
  }

  result.split_order.resize(N_splits);
  std::iota(result.split_order.begin(), result.split_order.end(), 0);

  result.parents_sorted.resize(C);
  std::iota(result.parents_sorted.begin(), result.parents_sorted.end(), 0);
  std::stable_sort(result.parents_sorted.begin(), result.parents_sorted.end(),
                   [&](int64_t a, int64_t b) {
                     if (result.split_counts[a] != result.split_counts[b]) {
                       return result.split_counts[a] > result.split_counts[b];
                     }
                     return a < b;
                   });

  std::vector<int64_t> parent_rank(C);
  for (size_t i = 0; i < C; ++i) {
    parent_rank[result.parents_sorted[i]] = static_cast<int64_t>(i);
  }

  std::stable_sort(result.split_order.begin(), result.split_order.end(),
                   [&](int64_t lhs, int64_t rhs) {
                     int64_t ra = parent_rank[split_parents[lhs]];
                     int64_t rb = parent_rank[split_parents[rhs]];
                     if (ra != rb) {
                       return ra < rb;
                     }
                     return lhs < rhs;
                   });

  result.split_parents_sorted.resize(N_splits);
  result.split_lefts_sorted.resize(N_splits);
  result.split_rights_sorted.resize(N_splits);
  result.log_split_probs_sorted.resize(N_splits);
  for (size_t i = 0; i < N_splits; ++i) {
    size_t idx = result.split_order[i];
    result.split_parents_sorted[i] = split_parents[idx];
    result.split_lefts_sorted[i] = split_lefts[idx];
    result.split_rights_sorted[i] = split_rights[idx];
    result.log_split_probs_sorted[i] = log_split_probs[idx];
  }

  result.split_leftrights_sorted.resize(2 * N_splits);
  for (size_t i = 0; i < N_splits; ++i) {
    result.split_leftrights_sorted[i] = result.split_lefts_sorted[i];
    result.split_leftrights_sorted[i + N_splits] = result.split_rights_sorted[i];
  }

  result.seg_counts.resize(C);
  for (size_t i = 0; i < C; ++i) {
    result.seg_counts[i] = result.split_counts[result.parents_sorted[i]];
  }

  result.ptr.resize(C + 1);
  result.ptr[0] = 0;
  for (size_t i = 0; i < C; ++i) {
    result.ptr[i + 1] = result.ptr[i] + result.seg_counts[i];
  }

  result.num_segs_ge2 = 0;
  result.num_segs_eq1 = 0;
  result.num_segs_eq0 = 0;
  for (int64_t count : result.seg_counts) {
    if (count >= 2) {
      ++result.num_segs_ge2;
    } else if (count == 1) {
      ++result.num_segs_eq1;
    } else {
      ++result.num_segs_eq0;
    }
  }

  result.stop_reduce_ptr_idx = result.num_segs_ge2;
  result.end_rows_ge2 = static_cast<int>(result.ptr[result.stop_reduce_ptr_idx]);
  result.ptr_ge2.assign(result.ptr.begin(), result.ptr.begin() + result.stop_reduce_ptr_idx + 1);

  // Compute clade inclusion DAG
  // Find ubiquitous clade (contains all leaves, has maximum size)
  int max_size = 0;
  result.ubiquitous_clade_id = -1;
  for (size_t i = 0; i < C; ++i) {
    const Clade& clade = clade_data.clades.get(i);
    if (clade.size() > max_size) {
      max_size = clade.size();
      result.ubiquitous_clade_id = static_cast<int64_t>(i);
    }
  }

  // Compute all inclusion relationships (child ⊆ parent)
  // For clades A and B: A ⊆ B iff (A.bits & B.bits) == A.bits
  for (size_t i = 0; i < C; ++i) {
    const Clade& clade_i = clade_data.clades.get(i);
    const BitVec& bits_i = clade_i.bits();

    for (size_t j = 0; j < C; ++j) {
      if (i == j) continue;  // Skip self-loops

      const Clade& clade_j = clade_data.clades.get(j);
      const BitVec& bits_j = clade_j.bits();

      // Check if clade_i ⊆ clade_j (all bits in i are also in j)
      bool is_subset = true;
      for (size_t w = 0; w < bits_i.size(); ++w) {
        if ((bits_i[w] & bits_j[w]) != bits_i[w]) {
          is_subset = false;
          break;
        }
      }

      if (is_subset) {
        result.inclusion_children.push_back(static_cast<int64_t>(i));
        result.inclusion_parents.push_back(static_cast<int64_t>(j));
      }
    }
  }

  return result;
}

/**
 * @brief Compute ancestors and recipients matrices for a species tree
 */
std::pair<std::vector<double>, std::vector<double>> compute_ancestors_and_recipients(
    TreeNode* species_root,
    int num_species) {

  std::vector<TreeNode*> species_postorder;
  collect_nodes_postorder(species_root, species_postorder);

  std::unordered_map<TreeNode*, int> species_node_to_index;
  species_node_to_index.reserve(species_postorder.size());
  for (size_t i = 0; i < species_postorder.size(); ++i) {
    species_node_to_index[species_postorder[i]] = static_cast<int>(i);
  }

  std::vector<double> ancestors(num_species * num_species, 0.0);
  for (TreeNode* node : species_postorder) {
    int idx = species_node_to_index[node];
    TreeNode* cur = node;
    while (cur) {
      int anc_idx = species_node_to_index[cur];
      ancestors[idx * num_species + anc_idx] = 1.0;
      cur = cur->parent;
    }
  }

  std::vector<double> recipients(num_species * num_species, 0.0);
  for (int i = 0; i < num_species; ++i) {
    double total = 0.0;
    for (int j = 0; j < num_species; ++j) {
      if (ancestors[i * num_species + j] == 0.0) {
        recipients[i * num_species + j] = 1.0;
        total += 1.0;
      }
    }
    if (total > 0.0) {
      for (int j = 0; j < num_species; ++j) {
        recipients[i * num_species + j] /= total;
      }
    }
  }

  return {std::move(ancestors), std::move(recipients)};
}

/**
 * @brief Amalgamate clades and splits from multiple gene trees
 *
 * Takes the union of all clades across gene trees. For each split,
 * the weight is the number of trees containing that split.
 */
CladeData amalgamate_clades_and_splits(
    const std::vector<std::string> &gene_paths,
    std::vector<std::string> &leaf_names,
    std::unordered_map<std::string, int> &leaf_to_index) {

  if (gene_paths.empty()) {
    throw std::runtime_error("No gene tree paths provided");
  }

  // Collect all unique leaf names from all trees
  std::unordered_set<std::string> all_leaves_set;
  std::vector<std::unique_ptr<TreeNode>> gene_trees;
  gene_trees.reserve(gene_paths.size());

  for (const std::string &path : gene_paths) {
    std::unique_ptr<TreeNode> tree = parse_newick_file(path);
    std::vector<std::string> tree_leaves;
    std::unordered_map<std::string, int> tree_leaf_map;
    collect_leaf_names(tree.get(), tree_leaves, tree_leaf_map);
    for (const std::string &name : tree_leaves) {
      all_leaves_set.insert(name);
    }
    gene_trees.push_back(std::move(tree));
  }

  // Create unified leaf ordering
  leaf_names.clear();
  leaf_names.reserve(all_leaves_set.size());
  for (const std::string &name : all_leaves_set) {
    leaf_names.push_back(name);
  }
  std::sort(leaf_names.begin(), leaf_names.end());

  leaf_to_index.clear();
  leaf_to_index.reserve(leaf_names.size());
  for (size_t i = 0; i < leaf_names.size(); ++i) {
    leaf_to_index[leaf_names[i]] = static_cast<int>(i);
  }

  const int num_leaves = static_cast<int>(leaf_names.size());
  if (num_leaves == 0) {
    throw std::runtime_error("No leaves found in gene trees");
  }

  CladeData result;
  const size_t num_words = bitvec_num_words(num_leaves);

  // Map from canonical split key to split index for weight accumulation
  std::map<std::tuple<int,int,int>, size_t> split_index_map;

  // Create root clade (all leaves)
  BitVec root_bits(num_words, 0ULL);
  for (int i = 0; i < num_leaves; ++i) {
    set_bit(root_bits, i);
  }
  result.root_clade_id = result.clades.get_or_create(std::move(root_bits));

  // Process each gene tree
  for (size_t tree_idx = 0; tree_idx < gene_trees.size(); ++tree_idx) {
    TreeNode *gene_root = gene_trees[tree_idx].get();

    std::vector<TreeNode *> postorder_nodes;
    collect_nodes_postorder(gene_root, postorder_nodes);
    const size_t num_nodes = postorder_nodes.size();

    std::unordered_map<TreeNode *, size_t> node_index;
    node_index.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      node_index[postorder_nodes[i]] = i;
    }

    std::vector<BitVec> node_clades(num_nodes, BitVec(num_words, 0ULL));
    std::vector<int> node_clade_ids(num_nodes, -1);
    std::vector<int> node_above_ids(num_nodes, -1);

    // Build clades for each node using global leaf indexing
    for (TreeNode *node : postorder_nodes) {
      size_t idx = node_index[node];
      BitVec bits(num_words, 0ULL);
      if (node->children.empty()) {
        int global_idx = leaf_to_index.at(node->name);
        set_bit(bits, global_idx);
      } else {
        const BitVec &left = node_clades[node_index[node->children[0]]];
        const BitVec &right = node_clades[node_index[node->children[1]]];
        for (size_t w = 0; w < num_words; ++w) {
          bits[w] = left[w] | right[w];
        }
      }
      node_clades[idx] = bits;
      node_clade_ids[idx] = result.clades.get_or_create(std::move(bits));
    }

    const BitVec &tree_root_bits = node_clades[node_index[gene_root]];
    int tree_root_id = node_clade_ids[node_index[gene_root]];

    std::unordered_set<PairKey, PairKeyHash, PairKeyEqual> tree_root_split_keys;

    // Compute "above" clades and root splits
    for (TreeNode *node : postorder_nodes) {
      if (node == gene_root) {
        continue;
      }
      size_t idx = node_index[node];
      const BitVec &below_bits = node_clades[idx];
      BitVec above_bits = bit_difference(tree_root_bits, below_bits);
      if (is_empty(above_bits)) {
        continue;
      }

      int below_id = node_clade_ids[idx];
      int above_id = result.clades.get_or_create(std::move(above_bits));
      node_above_ids[idx] = above_id;

      const Clade& below_clade = result.clades.get(below_id);
      const Clade& above_clade = result.clades.get(above_id);
      PairKey pkey = CladeSplit::canonical_root_key(below_id, below_clade, above_id, above_clade);

      if (tree_root_split_keys.insert(pkey).second) {
        CladeSplit split(tree_root_id, below_id, above_id, 1.0);
        auto key = split.canonical_key();
        auto it = split_index_map.find(key);
        if (it != split_index_map.end()) {
          result.splits[it->second].add_weight(1.0);
        } else {
          size_t new_idx = result.splits.size();
          result.splits.push_back(std::move(split));
          split_index_map[key] = new_idx;
        }
      }
    }

    // Create internal splits
    for (TreeNode *node : postorder_nodes) {
      if (node == gene_root || node->children.size() != 2) {
        continue;
      }
      size_t idx = node_index[node];
      int parent_id = node_clade_ids[idx];
      TreeNode *left_node = node->children[0];
      TreeNode *right_node = node->children[1];
      size_t left_idx = node_index[left_node];
      size_t right_idx = node_index[right_node];
      int left_id = node_clade_ids[left_idx];
      int right_id = node_clade_ids[right_idx];

      CladeSplit split(parent_id, left_id, right_id, 1.0);
      auto key = split.canonical_key();
      auto it = split_index_map.find(key);
      if (it != split_index_map.end()) {
        result.splits[it->second].add_weight(1.0);
      } else {
        size_t new_idx = result.splits.size();
        result.splits.push_back(std::move(split));
        split_index_map[key] = new_idx;
      }

      int above_id = node_above_ids[idx];
      if (above_id >= 0) {
        int left_plus_id = node_above_ids[right_idx];
        int right_plus_id = node_above_ids[left_idx];

        CladeSplit split1(left_plus_id, left_id, above_id, 1.0);
        auto key1 = split1.canonical_key();
        auto it1 = split_index_map.find(key1);
        if (it1 != split_index_map.end()) {
          result.splits[it1->second].add_weight(1.0);
        } else {
          size_t new_idx = result.splits.size();
          result.splits.push_back(std::move(split1));
          split_index_map[key1] = new_idx;
        }

        CladeSplit split2(right_plus_id, right_id, above_id, 1.0);
        auto key2 = split2.canonical_key();
        auto it2 = split_index_map.find(key2);
        if (it2 != split_index_map.end()) {
          result.splits[it2->second].add_weight(1.0);
        } else {
          size_t new_idx = result.splits.size();
          result.splits.push_back(std::move(split2));
          split_index_map[key2] = new_idx;
        }
      }
    }
  }

  return result;
}

/**
 * @brief Process multiple gene families, each with their own gene tree samples
 *
 * @param species_path Path to the species tree
 * @param families Map from family name to vector of gene tree paths for that family
 * @return Dictionary with shared species data and per-family CCPs
 */
py::dict preprocess_multiple_families(
    const std::string &species_path,
    const std::map<std::string, std::vector<std::string>> &families) {

  // Parse species tree once (shared across all families)
  std::unique_ptr<TreeNode> species_root = parse_newick_file(species_path);

  SpeciesData species_data;
  std::vector<TreeNode *> species_order;
  enumerate_species(species_root.get(), species_order, species_data);
  auto species_name_to_index = build_species_name_map(species_data);

  auto [ancestors, recipients] = compute_ancestors_and_recipients(species_root.get(), species_data.S);

  std::vector<int64_t> s_P_indexes;
  std::vector<int64_t> s_C1_indexes;
  std::vector<int64_t> s_C2_indexes;
  for (int i = 0; i < species_data.S; ++i) {
    if (species_data.children[i].size() == 2) {
      int left = species_data.children[i][0];
      int right = species_data.children[i][1];
      s_P_indexes.push_back(i);
      s_C1_indexes.push_back(left);
      s_C2_indexes.push_back(right);
    } else if (!species_data.children[i].empty()) {
      throw std::runtime_error("Species tree must be strictly binary");
    }
  }

  std::vector<int64_t> s_P_indexes_ext = s_P_indexes;
  for (int64_t idx : s_P_indexes) {
    s_P_indexes_ext.push_back(idx + species_data.S);
  }

  std::vector<int64_t> s_C12_indexes = s_C1_indexes;
  s_C12_indexes.insert(s_C12_indexes.end(), s_C2_indexes.begin(), s_C2_indexes.end());

  py::dict species_dict;
  species_dict["S"] = species_data.S;
  species_dict["names"] = species_data.names;
  species_dict["s_P_indexes"] = to_long_tensor(s_P_indexes_ext);
  species_dict["s_C12_indexes"] = to_long_tensor(s_C12_indexes);
  species_dict["ancestors_dense"] = to_double_matrix(ancestors, species_data.S, species_data.S);
  species_dict["Recipients_mat"] = to_double_matrix(recipients, species_data.S, species_data.S);
  species_dict["species_name_to_index"] = species_name_to_index;

  // Process each gene family
  py::dict families_dict;

  for (const auto& [family_name, gene_paths] : families) {
    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;

    CladeData clade_data = amalgamate_clades_and_splits(gene_paths, leaf_names, leaf_to_index);
    CCPArrays ccp = build_ccp_arrays(clade_data);

    const size_t C = clade_data.clades.size();

    // Build leaf-to-species mapping
    std::vector<int64_t> leaf_row_index;
    std::vector<int64_t> leaf_col_index;
    std::vector<std::vector<int64_t>> clade_leaves_indices(C);
    std::vector<std::string> clade_leaf_labels(C);
    std::vector<uint8_t> clade_is_leaf(C, 0);

    for (size_t cid = 0; cid < C; ++cid) {
      const Clade& clade = clade_data.clades.get(cid);
      const BitVec& bits = clade.bits();
      std::vector<int64_t> &indices = clade_leaves_indices[cid];
      for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
        uint64_t word = bits[word_index];
        while (word) {
          unsigned long bit = __builtin_ctzll(word);
          size_t leaf_idx = word_index * BITS_PER_WORD + bit;
          if (leaf_idx < leaf_names.size()) {
            indices.push_back(static_cast<int64_t>(leaf_idx));
            if (clade.size() == 1) {
              const std::string &leaf_name = leaf_names[leaf_idx];
              std::string species = extract_species_name(leaf_name);
              auto it = species_name_to_index.find(species);
              if (it == species_name_to_index.end()) {
                throw std::runtime_error("Species " + species +
                                         " not found for gene leaf " +
                                         leaf_name);
              }
              leaf_row_index.push_back(static_cast<int64_t>(cid));
              leaf_col_index.push_back(static_cast<int64_t>(it->second));
              clade_leaf_labels[cid] = leaf_name;
            }
          }
          word &= word - 1ULL;
        }
      }
      std::sort(indices.begin(), indices.end());
      if (clade.size() == 1) {
        clade_is_leaf[cid] = 1;
      }
    }

    py::dict ccp_dict;
    ccp_dict["clade_leaves"] = clade_leaves_indices;
    ccp_dict["clade_leaf_labels"] = clade_leaf_labels;
    ccp_dict["clade_is_leaf"] = to_uint8_tensor(clade_is_leaf);
    ccp_dict["split_counts"] = to_long_tensor(ccp.split_counts);
    ccp_dict["split_order"] = to_long_tensor(ccp.split_order);
    ccp_dict["split_parents_sorted"] = to_long_tensor(ccp.split_parents_sorted);
    ccp_dict["split_leftrights_sorted"] = to_long_tensor(ccp.split_leftrights_sorted);
    ccp_dict["log_split_probs_sorted"] = to_double_tensor(ccp.log_split_probs_sorted);
    ccp_dict["parents_sorted"] = to_long_tensor(ccp.parents_sorted);
    ccp_dict["seg_parent_ids"] = to_long_tensor(ccp.parents_sorted);
    ccp_dict["seg_counts"] = to_long_tensor(ccp.seg_counts);
    ccp_dict["ptr"] = to_long_tensor(ccp.ptr);
    ccp_dict["ptr_ge2"] = to_long_tensor(ccp.ptr_ge2);
    ccp_dict["num_segs_ge2"] = ccp.num_segs_ge2;
    ccp_dict["num_segs_eq1"] = ccp.num_segs_eq1;
    ccp_dict["num_segs_eq0"] = ccp.num_segs_eq0;
    ccp_dict["stop_reduce_ptr_idx"] = ccp.stop_reduce_ptr_idx;
    ccp_dict["end_rows_ge2"] = ccp.end_rows_ge2;
    ccp_dict["C"] = static_cast<int64_t>(C);
    ccp_dict["N_splits"] = static_cast<int64_t>(clade_data.splits.size());
    ccp_dict["root_clade_id"] = clade_data.root_clade_id;
    ccp_dict["inclusion_children"] = to_long_tensor(ccp.inclusion_children);
    ccp_dict["inclusion_parents"] = to_long_tensor(ccp.inclusion_parents);
    ccp_dict["ubiquitous_clade_id"] = ccp.ubiquitous_clade_id;

    py::dict family_dict;
    family_dict["ccp"] = ccp_dict;
    family_dict["root_clade_id"] = clade_data.root_clade_id;
    family_dict["leaf_row_index"] = to_long_tensor(leaf_row_index);
    family_dict["leaf_col_index"] = to_long_tensor(leaf_col_index);

    families_dict[family_name.c_str()] = family_dict;
  }

  py::dict result;
  result["species"] = species_dict;
  result["families"] = families_dict;

  return result;
}

py::dict preprocess(const std::string &species_path,
                    const std::vector<std::string> &gene_paths) {
  std::unique_ptr<TreeNode> species_root = parse_newick_file(species_path);

  std::vector<std::string> leaf_names;
  std::unordered_map<std::string, int> leaf_to_index;

  CladeData clade_data = amalgamate_clades_and_splits(gene_paths, leaf_names, leaf_to_index);
  CCPArrays ccp = build_ccp_arrays(clade_data);

  const size_t C = clade_data.clades.size();

  SpeciesData species_data;
  std::vector<TreeNode *> species_order;
  enumerate_species(species_root.get(), species_order, species_data);
  auto species_name_to_index = build_species_name_map(species_data);

  auto [ancestors, recipients] = compute_ancestors_and_recipients(species_root.get(), species_data.S);

  std::vector<int64_t> leaf_row_index;
  std::vector<int64_t> leaf_col_index;
  std::vector<std::vector<int64_t>> clade_leaves_indices(C);
  std::vector<std::string> clade_leaf_labels(C);
  std::vector<uint8_t> clade_is_leaf(C, 0);

  for (size_t cid = 0; cid < C; ++cid) {
    const Clade& clade = clade_data.clades.get(cid);
    const BitVec& bits = clade.bits();
    std::vector<int64_t> &indices = clade_leaves_indices[cid];
    for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
      uint64_t word = bits[word_index];
      while (word) {
        unsigned long bit = __builtin_ctzll(word);
        size_t leaf_idx = word_index * BITS_PER_WORD + bit;
        if (leaf_idx < leaf_names.size()) {
          indices.push_back(static_cast<int64_t>(leaf_idx));
          if (clade.size() == 1) {
            const std::string &leaf_name = leaf_names[leaf_idx];
            std::string species = extract_species_name(leaf_name);
            auto it = species_name_to_index.find(species);
            if (it == species_name_to_index.end()) {
              throw std::runtime_error("Species " + species +
                                       " not found for gene leaf " +
                                       leaf_name);
            }
            leaf_row_index.push_back(static_cast<int64_t>(cid));
            leaf_col_index.push_back(static_cast<int64_t>(it->second));
            clade_leaf_labels[cid] = leaf_name;
          }
        }
        word &= word - 1ULL;
      }
    }
    std::sort(indices.begin(), indices.end());
    if (clade.size() == 1) {
      clade_is_leaf[cid] = 1;
    }
  }

  py::dict species_dict;
  species_dict["S"] = species_data.S;
  species_dict["names"] = species_data.names;

  std::vector<int64_t> s_P_indexes;
  std::vector<int64_t> s_C1_indexes;
  std::vector<int64_t> s_C2_indexes;
  for (int i = 0; i < species_data.S; ++i) {
    if (species_data.children[i].size() == 2) {
      int left = species_data.children[i][0];
      int right = species_data.children[i][1];
      s_P_indexes.push_back(i);
      s_C1_indexes.push_back(left);
      s_C2_indexes.push_back(right);
    } else if (!species_data.children[i].empty()) {
      throw std::runtime_error("Species tree must be strictly binary");
    }
  }
  std::vector<int64_t> s_P_indexes_ext = s_P_indexes;
  for (int64_t idx : s_P_indexes) {
    s_P_indexes_ext.push_back(idx + species_data.S);
  }
  std::vector<int64_t> s_C12_indexes = s_C1_indexes;
  s_C12_indexes.insert(s_C12_indexes.end(), s_C2_indexes.begin(),
                       s_C2_indexes.end());

  species_dict["s_P_indexes"] = to_long_tensor(s_P_indexes_ext);
  species_dict["s_C12_indexes"] = to_long_tensor(s_C12_indexes);
  species_dict["ancestors_dense"] = to_double_matrix(ancestors, species_data.S, species_data.S);
  species_dict["Recipients_mat"] = to_double_matrix(recipients, species_data.S, species_data.S);
  species_dict["species_name_to_index"] = species_name_to_index;

  py::dict ccp_dict;
  ccp_dict["clade_leaves"] = clade_leaves_indices;
  ccp_dict["clade_leaf_labels"] = clade_leaf_labels;
  ccp_dict["clade_is_leaf"] = to_uint8_tensor(clade_is_leaf);
  ccp_dict["split_counts"] = to_long_tensor(ccp.split_counts);
  ccp_dict["split_order"] = to_long_tensor(ccp.split_order);
  ccp_dict["split_parents_sorted"] = to_long_tensor(ccp.split_parents_sorted);
  ccp_dict["split_leftrights_sorted"] = to_long_tensor(ccp.split_leftrights_sorted);
  ccp_dict["log_split_probs_sorted"] = to_double_tensor(ccp.log_split_probs_sorted);
  ccp_dict["parents_sorted"] = to_long_tensor(ccp.parents_sorted);
  ccp_dict["seg_parent_ids"] = to_long_tensor(ccp.parents_sorted);
  ccp_dict["seg_counts"] = to_long_tensor(ccp.seg_counts);
  ccp_dict["ptr"] = to_long_tensor(ccp.ptr);
  ccp_dict["ptr_ge2"] = to_long_tensor(ccp.ptr_ge2);
  ccp_dict["num_segs_ge2"] = ccp.num_segs_ge2;
  ccp_dict["num_segs_eq1"] = ccp.num_segs_eq1;
  ccp_dict["num_segs_eq0"] = ccp.num_segs_eq0;
  ccp_dict["stop_reduce_ptr_idx"] = ccp.stop_reduce_ptr_idx;
  ccp_dict["end_rows_ge2"] = ccp.end_rows_ge2;
  ccp_dict["C"] = static_cast<int64_t>(C);
  ccp_dict["N_splits"] = static_cast<int64_t>(clade_data.splits.size());
  ccp_dict["root_clade_id"] = clade_data.root_clade_id;
  ccp_dict["inclusion_children"] = to_long_tensor(ccp.inclusion_children);
  ccp_dict["inclusion_parents"] = to_long_tensor(ccp.inclusion_parents);
  ccp_dict["ubiquitous_clade_id"] = ccp.ubiquitous_clade_id;

  py::dict result;
  result["species"] = species_dict;
  result["ccp"] = ccp_dict;
  result["leaf_row_index"] = to_long_tensor(leaf_row_index);
  result["leaf_col_index"] = to_long_tensor(leaf_col_index);

  return result;
}

} // namespace

PYBIND11_MODULE(preprocess_cpp, m) {
  m.def("preprocess", &preprocess, "Preprocess species and gene trees");
  m.def("preprocess_multiple_families", &preprocess_multiple_families,
        "Preprocess multiple gene families with shared species tree");
}

PYBIND11_MODULE(preprocess_species_gene_cpp, n) {
  n.def("preprocess_species", &preprocess_species_tree, "Preprocess species tree only");
  n.def("preprocess_gene_with_species", &preprocess_gene_tree, "Preprocess gene tree with species helpers");
}
