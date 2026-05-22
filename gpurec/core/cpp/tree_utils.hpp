// Tree definitions and Newick parsing utilities
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct TreeNode {
  std::string name;
  std::vector<TreeNode *> children;
  TreeNode *parent{nullptr};
  size_t traversal_index{0};
  bool owns_children{true};

  TreeNode() = default;

  // Recursively delete all children to prevent memory leaks
  ~TreeNode() {
    if (owns_children) {
      for (TreeNode *child : children) {
        delete child;
      }
    }
  }

  // Prevent copying (raw pointer ownership makes copying unsafe)
  TreeNode(const TreeNode&) = delete;
  TreeNode& operator=(const TreeNode&) = delete;
  TreeNode(TreeNode&& other) noexcept
      : name(std::move(other.name)),
        children(std::move(other.children)),
        parent(other.parent),
        traversal_index(other.traversal_index),
        owns_children(other.owns_children) {
    other.owns_children = false;
  }
  TreeNode& operator=(TreeNode&& other) noexcept {
    if (this != &other) {
      if (owns_children) {
        for (TreeNode *child : children) {
          delete child;
        }
      }
      name = std::move(other.name);
      children = std::move(other.children);
      parent = other.parent;
      traversal_index = other.traversal_index;
      owns_children = other.owns_children;
      other.owns_children = false;
    }
    return *this;
  }
};

class NewickParser {
public:
  explicit NewickParser(std::string_view text);
  TreeNode *parse();

private:
  TreeNode *parse_subtree();
  void parse_leaf_label(TreeNode *node);
  void parse_node_label(TreeNode *node);
  void skip_branch_length();
  void skip_whitespace();
  static void trim(std::string &s);

  std::string_view text_;
  size_t pos_;
};

// Parse the retained simple-Newick subset: unquoted labels, optional numeric
// branch lengths, optional internal labels, no comments/metadata grammar.
// parse_newick_file expects one tree with optional terminal semicolon;
// parse_newick_trees_file accepts multiple semicolon-delimited records for
// gene-family CCP amalgamation and allows the final record to omit semicolon.
std::unique_ptr<TreeNode> parse_newick_file(const std::string &path);
std::vector<std::unique_ptr<TreeNode>> parse_newick_trees_file(const std::string &path);
void parse_newick_trees_file_into(const std::string &path,
                                  std::deque<TreeNode> &nodes,
                                  std::vector<TreeNode *> &roots);

// Post-order traversal collection
void collect_nodes_postorder(TreeNode *node, std::vector<TreeNode *> &order);

// Gene leaf helpers
void collect_leaf_names(TreeNode *node, std::vector<std::string> &leaf_names,
                        std::unordered_map<std::string, int> &leaf_to_idx);
void collect_leaf_names(TreeNode *node, std::unordered_set<std::string> &leaf_names);

// Species helpers
struct SpeciesData {
  int S;
  std::vector<std::string> names;
  std::vector<std::vector<int>> children;
};

void enumerate_species(TreeNode *root, std::vector<TreeNode *> &order,
                       SpeciesData &out);

std::unordered_map<std::string, int>
build_species_name_map(const SpeciesData &species);
