#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct TreeNode {
  std::string name;
  std::vector<TreeNode *> children;
  TreeNode *parent{nullptr};
};

class NewickParser {
public:
  explicit NewickParser(const std::string &text) : text_(text), pos_(0) {}

  TreeNode *parse() {
    TreeNode *root = parse_subtree();
    skip_whitespace();
    if (pos_ < text_.size() && text_[pos_] == ';') {
      ++pos_;
    }
    skip_whitespace();
    if (pos_ != text_.size()) {
      throw std::runtime_error("Unexpected trailing characters in Newick string");
    }
    return root;
  }

private:
  TreeNode *parse_subtree() {
    skip_whitespace();
    TreeNode *node = new TreeNode();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("Unexpected end of Newick string");
    }
    if (text_[pos_] == '(') {
      ++pos_;
      while (true) {
        TreeNode *child = parse_subtree();
        child->parent = node;
        node->children.push_back(child);
        skip_whitespace();
        if (pos_ >= text_.size()) {
          throw std::runtime_error("Unexpected end while parsing children");
        }
        char c = text_[pos_];
        if (c == ',') {
          ++pos_;
          continue;
        }
        if (c == ')') {
          ++pos_;
          break;
        }
        throw std::runtime_error("Expected ',' or ')' in Newick string");
      }
      parse_node_label(node);
    } else {
      parse_leaf_label(node);
    }
    skip_branch_length();
    return node;
  }

  void parse_leaf_label(TreeNode *node) {
    size_t start = pos_;
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c == ':' || c == ',' || c == ')' || c == '(' || c == ';') {
        break;
      }
      ++pos_;
    }
    node->name = text_.substr(start, pos_ - start);
    trim(node->name);
  }

  void parse_node_label(TreeNode *node) {
    skip_whitespace();
    size_t start = pos_;
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c == ':' || c == ',' || c == ')' || c == '(' || c == ';') {
        break;
      }
      ++pos_;
    }
    node->name = text_.substr(start, pos_ - start);
    trim(node->name);
  }

  void skip_branch_length() {
    skip_whitespace();
    if (pos_ < text_.size() && text_[pos_] == ':') {
      ++pos_;
      while (pos_ < text_.size()) {
        char c = text_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' ||
            c == 'e' || c == 'E' || c == '+' || c == '-') {
          ++pos_;
        } else {
          break;
        }
      }
    }
  }

  void skip_whitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  static void trim(std::string &s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    auto begin = std::find_if(s.begin(), s.end(), not_space);
    auto end = std::find_if(s.rbegin(), s.rend(), not_space).base();
    if (begin >= end) {
      s.clear();
    } else {
      s = std::string(begin, end);
    }
  }

  const std::string text_;
  size_t pos_;
};

std::unique_ptr<TreeNode> parse_newick_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Unable to open Newick file: " + path);
  }
  std::ostringstream buffer;
  buffer << f.rdbuf();
  std::string text = buffer.str();
  NewickParser parser(text);
  return std::unique_ptr<TreeNode>(parser.parse());
}

void collect_nodes_postorder(TreeNode *node, std::vector<TreeNode *> &order) {
  for (TreeNode *child : node->children) {
    collect_nodes_postorder(child, order);
  }
  order.push_back(node);
}

void collect_leaf_names(TreeNode *node, std::vector<std::string> &leaf_names,
                        std::unordered_map<std::string, int> &leaf_to_idx) {
  if (node->children.empty()) {
    auto it = leaf_to_idx.find(node->name);
    if (it == leaf_to_idx.end()) {
      int idx = static_cast<int>(leaf_names.size());
      leaf_to_idx[node->name] = idx;
      leaf_names.push_back(node->name);
    }
    return;
  }
  for (TreeNode *child : node->children) {
    collect_leaf_names(child, leaf_names, leaf_to_idx);
  }
}

using BitVec = std::vector<uint64_t>;

inline uint64_t wyhash64(uint64_t a, uint64_t b) {
  constexpr uint64_t secret0 = 0xa0761d6478bd642full;
  constexpr uint64_t secret1 = 0xe7037ed1a0b428dbull;
  a ^= secret0;
  b ^= secret1;
  a *= secret0;
  b *= secret1;
  return (a ^ (a >> 32)) + (b ^ (b >> 32));
}

struct BitVecHash {
  size_t operator()(const BitVec &bits) const noexcept {
    uint64_t hash = wyhash64(static_cast<uint64_t>(bits.size()),
                             0x8ebc6af09c88c6e3ULL);
    for (uint64_t word : bits) {
      hash = wyhash64(hash, word);
    }
    return static_cast<size_t>(hash);
  }
};

struct BitVecEqual {
  bool operator()(const BitVec &a, const BitVec &b) const noexcept {
    return a == b;
  }
};

inline BitVec make_bitvec(size_t words) { return BitVec(words, 0ULL); }

inline void set_bit(BitVec &bits, int index) {
  size_t w = static_cast<size_t>(index) >> 6;
  size_t offset = static_cast<size_t>(index) & 63ULL;
  bits[w] |= (1ULL << offset);
}

inline BitVec bit_or(const BitVec &a, const BitVec &b) {
  BitVec result(a.size(), 0ULL);
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] | b[i];
  }
  return result;
}

inline BitVec bit_difference(const BitVec &parent, const BitVec &child) {
  BitVec result(parent.size(), 0ULL);
  for (size_t i = 0; i < parent.size(); ++i) {
    result[i] = parent[i] & (~child[i]);
  }
  return result;
}

inline int bit_count(const BitVec &bits) {
  int count = 0;
  for (uint64_t word : bits) {
    count += static_cast<int>(__builtin_popcountll(word));
  }
  return count;
}

inline bool is_empty(const BitVec &bits) {
  for (uint64_t word : bits) {
    if (word != 0ULL) {
      return false;
    }
  }
  return true;
}

inline bool bitvec_lex_less(const BitVec &a, const BitVec &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

struct PairKey {
  int first;
  int second;
  PairKey(int a, int b) : first(a), second(b) {}
};

struct PairKeyHash {
  size_t operator()(const PairKey &key) const noexcept {
    uint64_t a = static_cast<uint32_t>(key.first);
    uint64_t b = static_cast<uint32_t>(key.second);
    return static_cast<size_t>(wyhash64(a, b));
  }
};

struct PairKeyEqual {
  bool operator()(const PairKey &a, const PairKey &b) const noexcept {
    return a.first == b.first && a.second == b.second;
  }
};

struct SpeciesData {
  int S;
  std::vector<std::string> names;
  std::vector<std::vector<int>> children;
};

void enumerate_species(TreeNode *root, std::vector<TreeNode *> &order,
                       SpeciesData &out) {
  collect_nodes_postorder(root, order);
  int S = static_cast<int>(order.size());
  std::unordered_map<TreeNode *, int> index;
  out.names.resize(S);
  out.children.resize(S);
  for (int i = 0; i < S; ++i) {
    index[order[i]] = i;
  }
  for (int i = 0; i < S; ++i) {
    TreeNode *node = order[i];
    out.names[i] = node->name;
    for (TreeNode *child : node->children) {
      out.children[i].push_back(index[child]);
    }
  }
  out.S = S;
}

std::unordered_map<std::string, int>
build_species_name_map(const SpeciesData &species) {
  std::unordered_map<std::string, int> mapping;
  for (int i = 0; i < species.S; ++i) {
    if (!species.names[i].empty()) {
      mapping[species.names[i]] = i;
    }
  }
  return mapping;
}

struct SplitRecord {
  int parent;
  int left;
  int right;
  double weight;
};

std::string extract_species_name(const std::string &leaf_name) {
  auto pos = leaf_name.find('_');
  if (pos != std::string::npos) {
    return leaf_name.substr(0, pos);
  }
  if (leaf_name.size() > 1 && leaf_name[0] == 'n') {
    size_t i = 1;
    while (i < leaf_name.size() &&
           std::isdigit(static_cast<unsigned char>(leaf_name[i]))) {
      ++i;
    }
    return leaf_name.substr(1, i - 1);
  }
  return leaf_name;
}

py::dict preprocess(const std::string &species_path,
                    const std::string &gene_path) {
  auto species_root = parse_newick_file(species_path);
  auto gene_root = parse_newick_file(gene_path);

  std::vector<std::string> leaf_names;
  std::unordered_map<std::string, int> leaf_to_index;
  collect_leaf_names(gene_root.get(), leaf_names, leaf_to_index);
  const int num_leaves = static_cast<int>(leaf_names.size());
  if (num_leaves == 0) {
    throw std::runtime_error("Gene tree has no leaves");
  }
  const size_t num_words = (static_cast<size_t>(num_leaves) + 63ULL) >> 6;

  std::vector<TreeNode *> postorder_nodes;
  collect_nodes_postorder(gene_root.get(), postorder_nodes);
  const size_t num_nodes = postorder_nodes.size();

  std::unordered_map<TreeNode *, size_t> node_index;
  node_index.reserve(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    node_index[postorder_nodes[i]] = i;
  }

  std::vector<BitVec> node_clades(num_nodes, BitVec(num_words, 0ULL));
  std::vector<int> node_clade_ids(num_nodes, -1);

  std::unordered_map<BitVec, int, BitVecHash, BitVecEqual> clade_to_id;
  clade_to_id.reserve(num_nodes * 2);
  std::vector<BitVec> id_to_clade;
  id_to_clade.reserve(num_nodes * 2);
  std::vector<int> clade_sizes;

  auto get_or_add = [&](const BitVec &bits) {
    auto it = clade_to_id.find(bits);
    if (it != clade_to_id.end()) {
      return it->second;
    }
    BitVec stored = bits;
    int size = bit_count(stored);
    int new_id = static_cast<int>(id_to_clade.size());
    id_to_clade.push_back(std::move(stored));
    clade_sizes.push_back(size);
    clade_to_id.emplace(id_to_clade.back(), new_id);
    return new_id;
  };

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
    node_clade_ids[idx] = get_or_add(node_clades[idx]);
  }

  const BitVec &root_bits = node_clades[node_index[gene_root.get()]];
  const int root_clade_id = node_clade_ids[node_index[gene_root.get()]];

  std::vector<SplitRecord> splits;
  splits.reserve(num_nodes * 3);
  std::unordered_set<PairKey, PairKeyHash, PairKeyEqual> root_split_keys;
  root_split_keys.reserve(num_nodes * 2);

  auto ensure_clade = [&](const BitVec &bits) {
    auto it = clade_to_id.find(bits);
    if (it != clade_to_id.end()) {
      return it->second;
    }
    BitVec stored = bits;
    int size = bit_count(stored);
    int new_id = static_cast<int>(id_to_clade.size());
    id_to_clade.push_back(std::move(stored));
    clade_sizes.push_back(size);
    clade_to_id.emplace(id_to_clade.back(), new_id);
    return new_id;
  };

  for (TreeNode *node : postorder_nodes) {
    if (node == gene_root.get()) {
      continue;
    }
    size_t idx = node_index[node];
    const BitVec &below_bits = node_clades[idx];
    BitVec above_bits = bit_difference(root_bits, below_bits);
    if (is_empty(above_bits)) {
      continue;
    }

    int below_id = node_clade_ids[idx];
    int above_id = ensure_clade(above_bits);
    const BitVec &above_clade = id_to_clade[static_cast<size_t>(above_id)];

    auto canonical_key = [&](int a_id, const BitVec &a_bits, int b_id,
                             const BitVec &b_bits) {
      size_t a_idx = static_cast<size_t>(a_id);
      size_t b_idx = static_cast<size_t>(b_id);
      if (clade_sizes[a_idx] < clade_sizes[b_idx]) {
        return PairKey{a_id, b_id};
      }
      if (clade_sizes[a_idx] > clade_sizes[b_idx]) {
        return PairKey{b_id, a_id};
      }
      if (bitvec_lex_less(a_bits, b_bits)) {
        return PairKey{a_id, b_id};
      }
      if (bitvec_lex_less(b_bits, a_bits)) {
        return PairKey{b_id, a_id};
      }
      int first = std::min(a_id, b_id);
      int second = std::max(a_id, b_id);
      return PairKey{first, second};
    };

    PairKey key = canonical_key(below_id, below_bits, above_id, above_clade);
    if (root_split_keys.insert(key).second) {
      splits.push_back({root_clade_id, below_id, above_id, 1.0});
    }
  }

  for (TreeNode *node : postorder_nodes) {
    if (node == gene_root.get()) {
      continue;
    }
    if (node->children.size() != 2) {
      continue;
    }
    size_t idx = node_index[node];
    int parent_id = node_clade_ids[idx];
    TreeNode *left_node = node->children[0];
    TreeNode *right_node = node->children[1];
    int left_id = node_clade_ids[node_index[left_node]];
    int right_id = node_clade_ids[node_index[right_node]];
    splits.push_back({parent_id, left_id, right_id, 1.0});

    BitVec above_bits = bit_difference(root_bits, node_clades[idx]);
    if (!is_empty(above_bits)) {
      int above_id = ensure_clade(above_bits);
      BitVec left_plus = bit_or(node_clades[node_index[left_node]], above_bits);
      int left_plus_id = ensure_clade(left_plus);
      splits.push_back({left_plus_id, left_id, above_id, 1.0});

      BitVec right_plus = bit_or(node_clades[node_index[right_node]], above_bits);
      int right_plus_id = ensure_clade(right_plus);
      splits.push_back({right_plus_id, right_id, above_id, 1.0});
    }
  }

  const size_t C = id_to_clade.size();
  const size_t N_splits = splits.size();

  std::vector<int64_t> split_parents(N_splits);
  std::vector<int64_t> split_lefts(N_splits);
  std::vector<int64_t> split_rights(N_splits);
  std::vector<double> split_weights(N_splits);
  std::vector<int64_t> split_counts(C, 0);

  for (size_t i = 0; i < N_splits; ++i) {
    split_parents[i] = splits[i].parent;
    split_lefts[i] = splits[i].left;
    split_rights[i] = splits[i].right;
    split_weights[i] = splits[i].weight;
    split_counts[splits[i].parent] += 1;
  }

  std::vector<double> sum_weights(C, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    sum_weights[split_parents[i]] += split_weights[i];
  }

  std::vector<double> split_probs(N_splits, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    double denom = sum_weights[split_parents[i]];
    split_probs[i] = denom > 0.0 ? split_weights[i] / denom : 0.0;
  }

  std::vector<double> log_split_probs(N_splits, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    log_split_probs[i] = split_probs[i] > 0.0
                              ? std::log(split_probs[i])
                              : -std::numeric_limits<double>::infinity();
  }

  std::vector<int64_t> split_order(N_splits);
  std::iota(split_order.begin(), split_order.end(), 0);

  std::vector<int64_t> parents_sorted(C);
  std::iota(parents_sorted.begin(), parents_sorted.end(), 0);
  std::stable_sort(parents_sorted.begin(), parents_sorted.end(),
                   [&](int64_t a, int64_t b) {
                     if (split_counts[a] != split_counts[b]) {
                       return split_counts[a] > split_counts[b];
                     }
                     return a < b;
                   });
  std::vector<int64_t> parent_rank(C);
  for (size_t i = 0; i < C; ++i) {
    parent_rank[parents_sorted[i]] = static_cast<int64_t>(i);
  }
  std::stable_sort(split_order.begin(), split_order.end(),
                   [&](int64_t lhs, int64_t rhs) {
                     int64_t ra = parent_rank[split_parents[lhs]];
                     int64_t rb = parent_rank[split_parents[rhs]];
                     if (ra != rb) {
                       return ra < rb;
                     }
                     return lhs < rhs;
                   });

  std::vector<int64_t> split_parents_sorted(N_splits);
  std::vector<int64_t> split_lefts_sorted(N_splits);
  std::vector<int64_t> split_rights_sorted(N_splits);
  std::vector<double> log_split_probs_sorted(N_splits);
  for (size_t i = 0; i < N_splits; ++i) {
    size_t idx = split_order[i];
    split_parents_sorted[i] = split_parents[idx];
    split_lefts_sorted[i] = split_lefts[idx];
    split_rights_sorted[i] = split_rights[idx];
    log_split_probs_sorted[i] = log_split_probs[idx];
  }
  std::vector<int64_t> split_leftrights_sorted(2 * N_splits);
  for (size_t i = 0; i < N_splits; ++i) {
    split_leftrights_sorted[i] = split_lefts_sorted[i];
    split_leftrights_sorted[i + N_splits] = split_rights_sorted[i];
  }

  std::vector<int64_t> seg_counts(C);
  for (size_t i = 0; i < C; ++i) {
    seg_counts[i] = split_counts[parents_sorted[i]];
  }
  std::vector<int64_t> ptr(C + 1);
  ptr[0] = 0;
  for (size_t i = 0; i < C; ++i) {
    ptr[i + 1] = ptr[i] + seg_counts[i];
  }
  int num_segs_ge2 = 0;
  int num_segs_eq1 = 0;
  int num_segs_eq0 = 0;
  for (int64_t count : seg_counts) {
    if (count >= 2) {
      ++num_segs_ge2;
    } else if (count == 1) {
      ++num_segs_eq1;
    } else {
      ++num_segs_eq0;
    }
  }
  int stop_reduce_ptr_idx = num_segs_ge2;
  int end_rows_ge2 = static_cast<int>(ptr[stop_reduce_ptr_idx]);
  std::vector<int64_t> ptr_ge2(ptr.begin(), ptr.begin() + stop_reduce_ptr_idx + 1);

  SpeciesData species_data;
  std::vector<TreeNode *> species_order;
  enumerate_species(species_root.get(), species_order, species_data);
  auto species_name_to_index = build_species_name_map(species_data);

  torch::TensorOptions long_opts = torch::TensorOptions().dtype(torch::kInt64);

  auto to_long_tensor = [&](const std::vector<int64_t> &vec) {
    return torch::from_blob(const_cast<int64_t *>(vec.data()),
                            {static_cast<long>(vec.size())}, long_opts)
        .clone();
  };
  auto to_double_tensor = [&](const std::vector<double> &vec) {
    return torch::from_blob(const_cast<double *>(vec.data()),
                            {static_cast<long>(vec.size())},
                            torch::TensorOptions().dtype(torch::kFloat64))
        .clone();
  };

  torch::Tensor split_parents_tensor = to_long_tensor(split_parents);
  torch::Tensor split_lefts_tensor = to_long_tensor(split_lefts);
  torch::Tensor split_rights_tensor = to_long_tensor(split_rights);
  torch::Tensor log_split_probs_tensor = to_double_tensor(log_split_probs);
  torch::Tensor split_counts_tensor = to_long_tensor(split_counts);
  torch::Tensor split_order_tensor = to_long_tensor(split_order);
  torch::Tensor split_parents_sorted_tensor = to_long_tensor(split_parents_sorted);
  torch::Tensor split_lefts_sorted_tensor = to_long_tensor(split_lefts_sorted);
  torch::Tensor split_rights_sorted_tensor = to_long_tensor(split_rights_sorted);
  torch::Tensor split_leftrights_sorted_tensor = to_long_tensor(split_leftrights_sorted);
  torch::Tensor log_split_probs_sorted_tensor = to_double_tensor(log_split_probs_sorted);
  torch::Tensor parents_sorted_tensor = to_long_tensor(parents_sorted);
  torch::Tensor seg_parent_ids_tensor = to_long_tensor(parents_sorted);
  torch::Tensor seg_counts_tensor = to_long_tensor(seg_counts);
  torch::Tensor ptr_tensor = to_long_tensor(ptr);
  torch::Tensor ptr_ge2_tensor = to_long_tensor(ptr_ge2);

  torch::Tensor clade_species_tensor = torch::zeros(
      {static_cast<long>(C), species_data.S},
      torch::TensorOptions().dtype(torch::kFloat64));
  auto tensor_accessor = clade_species_tensor.accessor<double, 2>();

  std::vector<std::vector<int64_t>> clade_leaves_indices(C);
  std::vector<std::string> clade_leaf_labels(C, std::string());
  std::vector<uint8_t> clade_is_leaf(C, 0);
  for (size_t cid = 0; cid < C; ++cid) {
    const BitVec &bits = id_to_clade[cid];
    auto &indices = clade_leaves_indices[cid];
    for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
      uint64_t word = bits[word_index];
      while (word) {
        unsigned long bit = __builtin_ctzll(word);
        size_t leaf_idx = word_index * 64 + bit;
        if (leaf_idx < leaf_names.size()) {
          indices.push_back(static_cast<int64_t>(leaf_idx));
          if (clade_sizes[cid] == 1) {
            const std::string &leaf_name = leaf_names[leaf_idx];
            std::string species = extract_species_name(leaf_name);
            auto it = species_name_to_index.find(species);
            if (it == species_name_to_index.end()) {
              throw std::runtime_error("Species " + species +
                                       " not found for gene leaf " +
                                       leaf_name);
            }
            tensor_accessor[static_cast<long>(cid)][it->second] = 1.0;
            clade_leaf_labels[cid] = leaf_name; // store original leaf label
          }
        }
        word &= word - 1ULL;
      }
    }
    std::sort(indices.begin(), indices.end());
    if (clade_sizes[cid] == 1) {
      clade_is_leaf[cid] = 1;
    }
  }

  py::dict species_dict;
  species_dict["S"] = species_data.S;
  species_dict["names"] = species_data.names;

  std::vector<double> s_C1(species_data.S * species_data.S, 0.0);
  std::vector<double> s_C2(species_data.S * species_data.S, 0.0);
  std::vector<int64_t> s_P_indexes;
  std::vector<int64_t> s_C1_indexes;
  std::vector<int64_t> s_C2_indexes;
  for (int i = 0; i < species_data.S; ++i) {
    if (species_data.children[i].size() == 2) {
      int left = species_data.children[i][0];
      int right = species_data.children[i][1];
      s_C1[i * species_data.S + left] = 1.0;
      s_C2[i * species_data.S + right] = 1.0;
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

  std::vector<double> ancestors(species_data.S * species_data.S, 0.0);
  std::vector<TreeNode *> species_postorder;
  collect_nodes_postorder(species_root.get(), species_postorder);
  std::unordered_map<TreeNode *, int> species_node_to_index;
  species_node_to_index.reserve(species_postorder.size());
  for (size_t i = 0; i < species_postorder.size(); ++i) {
    species_node_to_index[species_postorder[i]] = static_cast<int>(i);
  }
  for (TreeNode *node : species_postorder) {
    int idx = species_node_to_index[node];
    TreeNode *cur = node;
    while (cur) {
      int anc_idx = species_node_to_index[cur];
      ancestors[idx * species_data.S + anc_idx] = 1.0;
      cur = cur->parent;
    }
  }

  std::vector<double> recipients(species_data.S * species_data.S, 0.0);
  for (int i = 0; i < species_data.S; ++i) {
    double total = 0.0;
    for (int j = 0; j < species_data.S; ++j) {
      if (ancestors[i * species_data.S + j] == 0.0) {
        recipients[i * species_data.S + j] = 1.0;
        total += 1.0;
      }
    }
    if (total > 0.0) {
      for (int j = 0; j < species_data.S; ++j) {
        recipients[i * species_data.S + j] /= total;
      }
    }
  }

  auto to_double_matrix = [&](const std::vector<double> &mat, int rows,
                              int cols) {
    return torch::from_blob(const_cast<double *>(mat.data()), {rows, cols},
                            torch::TensorOptions().dtype(torch::kFloat64))
        .clone();
  };

  species_dict["s_C1"] =
      to_double_matrix(s_C1, species_data.S, species_data.S);
  species_dict["s_C2"] =
      to_double_matrix(s_C2, species_data.S, species_data.S);
  species_dict["s_P_indexes"] =
      torch::from_blob(const_cast<int64_t *>(s_P_indexes_ext.data()),
                       {static_cast<long>(s_P_indexes_ext.size())}, long_opts)
          .clone();
  species_dict["s_C1_indexes"] =
      torch::from_blob(const_cast<int64_t *>(s_C1_indexes.data()),
                       {static_cast<long>(s_C1_indexes.size())}, long_opts)
          .clone();
  species_dict["s_C2_indexes"] =
      torch::from_blob(const_cast<int64_t *>(s_C2_indexes.data()),
                       {static_cast<long>(s_C2_indexes.size())}, long_opts)
          .clone();
  species_dict["s_C12_indexes"] =
      torch::from_blob(const_cast<int64_t *>(s_C12_indexes.data()),
                       {static_cast<long>(s_C12_indexes.size())}, long_opts)
          .clone();
  species_dict["ancestors_dense"] =
      to_double_matrix(ancestors, species_data.S, species_data.S);
  species_dict["Recipients_mat"] =
      to_double_matrix(recipients, species_data.S, species_data.S);

  std::vector<uint8_t> sp_leaves_mask(species_data.S, 0);
  std::vector<uint8_t> sp_internal_mask(species_data.S, 0);
  for (int i = 0; i < species_data.S; ++i) {
    if (species_data.children[i].empty()) {
      sp_leaves_mask[i] = 1;
    } else {
      sp_internal_mask[i] = 1;
    }
  }

  auto to_bool_tensor = [&](const std::vector<uint8_t> &vec) {
    return torch::from_blob(const_cast<uint8_t *>(vec.data()),
                            {static_cast<long>(vec.size())},
                            torch::TensorOptions().dtype(torch::kUInt8))
        .clone();
  };


  species_dict["species_name_to_index"] = species_name_to_index;

  py::dict ccp_dict;
  ccp_dict["clade_leaves"] = clade_leaves_indices;
  // Provide optional labels for leaf clades (empty for non-leaf)
  ccp_dict["clade_leaf_labels"] = clade_leaf_labels;
  // Optional mask for leaf clades
  ccp_dict["clade_is_leaf"] = torch::from_blob(
                                   clade_is_leaf.data(),
                                   {static_cast<long>(C)},
                                   torch::TensorOptions().dtype(torch::kUInt8))
                                   .clone();


  ccp_dict["split_counts"] = split_counts_tensor;
  ccp_dict["split_order"] = split_order_tensor;
  ccp_dict["split_parents_sorted"] = split_parents_sorted_tensor;
  ccp_dict["split_leftrights_sorted"] = split_leftrights_sorted_tensor;
  ccp_dict["log_split_probs_sorted"] = log_split_probs_sorted_tensor;
  ccp_dict["parents_sorted"] = parents_sorted_tensor;
  ccp_dict["seg_parent_ids"] = seg_parent_ids_tensor;
  ccp_dict["seg_counts"] = seg_counts_tensor;
  ccp_dict["ptr"] = ptr_tensor;
  ccp_dict["ptr_ge2"] = ptr_ge2_tensor;
  ccp_dict["num_segs_ge2"] = num_segs_ge2;
  ccp_dict["num_segs_eq1"] = num_segs_eq1;
  ccp_dict["num_segs_eq0"] = num_segs_eq0;
  ccp_dict["stop_reduce_ptr_idx"] = stop_reduce_ptr_idx;
  ccp_dict["end_rows_ge2"] = end_rows_ge2;
  ccp_dict["C"] = static_cast<int64_t>(C);
  ccp_dict["N_splits"] = static_cast<int64_t>(N_splits);
  ccp_dict["root_clade_id"] = root_clade_id;

  py::dict result;
  result["species"] = species_dict;
  result["ccp"] = ccp_dict;
  result["clade_species_map"] = clade_species_tensor;
  return result;
}

} // namespace

PYBIND11_MODULE(preprocess_cpp, m) {
  m.def("preprocess", &preprocess, "Preprocess species and gene trees");
}
