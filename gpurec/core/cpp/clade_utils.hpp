// Clade hashing/comparison and split key utilities
#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

using BitVec = std::vector<uint64_t>;

uint64_t wyhash64(uint64_t a, uint64_t b);

struct BitVecHash {
  size_t operator()(const BitVec &bits) const noexcept;
};

struct BitVecEqual {
  bool operator()(const BitVec &a, const BitVec &b) const noexcept;
};

BitVec make_bitvec(size_t words);
void set_bit(BitVec &bits, int index);
BitVec bit_or(const BitVec &a, const BitVec &b);
BitVec bit_difference(const BitVec &parent, const BitVec &child);
int bit_count(const BitVec &bits);
bool is_empty(const BitVec &bits);
bool bitvec_lex_less(const BitVec &a, const BitVec &b);

struct PairKey {
  int first;
  int second;
  PairKey(int a, int b) : first(a), second(b) {}
};

struct PairKeyHash {
  size_t operator()(const PairKey &key) const noexcept;
};

struct PairKeyEqual {
  bool operator()(const PairKey &a, const PairKey &b) const noexcept;
};

struct SplitRecord {
  int parent;
  int left;
  int right;
  double weight;
};

