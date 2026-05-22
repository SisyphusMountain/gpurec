#include "tree_utils.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

std::string_view trim_view(std::string_view s) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  auto begin = std::find_if(s.begin(), s.end(), not_space);
  auto end = std::find_if(s.rbegin(), s.rend(), not_space).base();
  if (begin >= end) {
    return {};
  }
  return std::string_view(&*begin, static_cast<size_t>(end - begin));
}

void trim_string(std::string &s) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  auto begin = std::find_if(s.begin(), s.end(), not_space);
  auto end = std::find_if(s.rbegin(), s.rend(), not_space).base();
  if (begin >= end) {
    s.clear();
  } else {
    s = std::string(begin, end);
  }
}

std::string read_text_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Unable to open Newick file: " + path);
  }
  std::ostringstream buffer;
  buffer << f.rdbuf();
  return buffer.str();
}

class ArenaNewickParser {
public:
  ArenaNewickParser(std::string_view text, std::deque<TreeNode> &nodes)
      : text_(text), nodes_(nodes), pos_(0) {}

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
  TreeNode *make_node() {
    nodes_.emplace_back();
    TreeNode *node = &nodes_.back();
    node->owns_children = false;
    return node;
  }

  TreeNode *parse_subtree() {
    skip_whitespace();
    TreeNode *node = make_node();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("Unexpected end of Newick string");
    }
    if (text_[pos_] == '(') {
      ++pos_;
      node->children.reserve(2);
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
    node->name = std::string(text_.substr(start, pos_ - start));
    trim_string(node->name);
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
    node->name = std::string(text_.substr(start, pos_ - start));
    trim_string(node->name);
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

  std::string_view text_;
  std::deque<TreeNode> &nodes_;
  size_t pos_;
};

}  // namespace

NewickParser::NewickParser(std::string_view text) : text_(text), pos_(0) {}

TreeNode *NewickParser::parse() {
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

TreeNode *NewickParser::parse_subtree() {
  skip_whitespace();
  TreeNode *node = new TreeNode();
  if (pos_ >= text_.size()) {
    throw std::runtime_error("Unexpected end of Newick string");
  }
  if (text_[pos_] == '(') {
    ++pos_;
    node->children.reserve(2);
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

void NewickParser::parse_leaf_label(TreeNode *node) {
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

void NewickParser::parse_node_label(TreeNode *node) {
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

void NewickParser::skip_branch_length() {
  skip_whitespace();
  if (pos_ < text_.size() && text_[pos_] == ':') {
    ++pos_;
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
  }
}

void NewickParser::skip_whitespace() {
  while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
    ++pos_;
  }
}

void NewickParser::trim(std::string &s) {
  trim_string(s);
}

std::unique_ptr<TreeNode> parse_newick_file(const std::string &path) {
  std::string text = read_text_file(path);
  NewickParser parser(text);
  return std::unique_ptr<TreeNode>(parser.parse());
}

std::vector<std::unique_ptr<TreeNode>> parse_newick_trees_file(const std::string &path) {
  std::string text = read_text_file(path);
  std::vector<std::unique_ptr<TreeNode>> trees;
  size_t start = 0;
  while (start < text.size()) {
    size_t semi = text.find(';', start);
    std::string_view record;
    if (semi == std::string::npos) {
      record = trim_view(std::string_view(text).substr(start));
      start = text.size();
    } else {
      record = trim_view(std::string_view(text).substr(start, semi - start + 1));
      start = semi + 1;
    }
    if (record.empty() || record == ";") {
      continue;
    }
    NewickParser parser(record);
    trees.emplace_back(parser.parse());
  }
  if (trees.empty()) {
    throw std::runtime_error("No Newick trees found in file: " + path);
  }
  return trees;
}

void parse_newick_trees_file_into(const std::string &path,
                                  std::deque<TreeNode> &nodes,
                                  std::vector<TreeNode *> &roots) {
  std::string text = read_text_file(path);
  const size_t roots_before = roots.size();
  size_t start = 0;
  while (start < text.size()) {
    size_t semi = text.find(';', start);
    std::string_view record;
    if (semi == std::string::npos) {
      record = trim_view(std::string_view(text).substr(start));
      start = text.size();
    } else {
      record = trim_view(std::string_view(text).substr(start, semi - start + 1));
      start = semi + 1;
    }
    if (record.empty() || record == ";") {
      continue;
    }
    ArenaNewickParser parser(record, nodes);
    roots.push_back(parser.parse());
  }
  if (roots.size() == roots_before) {
    throw std::runtime_error("No Newick trees found in file: " + path);
  }
}

void collect_nodes_postorder(TreeNode *node, std::vector<TreeNode *> &order) {
  for (TreeNode *child : node->children) {
    collect_nodes_postorder(child, order);
  }
  node->traversal_index = order.size();
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

void collect_leaf_names(TreeNode *node, std::unordered_set<std::string> &leaf_names) {
  if (node->children.empty()) {
    leaf_names.insert(node->name);
    return;
  }
  for (TreeNode *child : node->children) {
    collect_leaf_names(child, leaf_names);
  }
}

void enumerate_species(TreeNode *root, std::vector<TreeNode *> &order, SpeciesData &out) {
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

std::unordered_map<std::string, int> build_species_name_map(const SpeciesData &species) {
  std::unordered_map<std::string, int> mapping;
  for (int i = 0; i < species.S; ++i) {
    if (!species.names[i].empty()) {
      mapping[species.names[i]] = i;
    }
  }
  return mapping;
}
