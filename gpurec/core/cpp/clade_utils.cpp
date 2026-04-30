#include "clade_utils.hpp"

#include <algorithm>

// Hash clades recursively using the two subclade hashes
uint64_t wyhash64(uint64_t a, uint64_t b) {
  constexpr uint64_t secret0 = 0xa0761d6478bd642full;
  constexpr uint64_t secret1 = 0xe7037ed1a0b428dbull;
  a ^= secret0;
  b ^= secret1;
  a *= secret0;
  b *= secret1;
  return (a ^ (a >> 32)) + (b ^ (b >> 32));
}

size_t BitVecHash::operator()(const BitVec &bits) const noexcept {
  uint64_t hash = wyhash64(static_cast<uint64_t>(bits.size()), 0x8ebc6af09c88c6e3ULL);
  for (uint64_t word : bits) {
    hash = wyhash64(hash, word);
  }
  return static_cast<size_t>(hash);
}

bool BitVecEqual::operator()(const BitVec &a, const BitVec &b) const noexcept {
  return a == b;
}

BitVec make_bitvec(size_t words) { return BitVec(words, 0ULL); }

void set_bit(BitVec &bits, int index) {
  size_t w = static_cast<size_t>(index) >> 6;
  size_t offset = static_cast<size_t>(index) & 63ULL;
  bits[w] |= (1ULL << offset);
}

BitVec bit_or(const BitVec &a, const BitVec &b) {
  BitVec result(a.size(), 0ULL);
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] | b[i];
  }
  return result;
}

BitVec bit_difference(const BitVec &parent, const BitVec &child) {
  BitVec result(parent.size(), 0ULL);
  for (size_t i = 0; i < parent.size(); ++i) {
    result[i] = parent[i] & (~child[i]);
  }
  return result;
}

int bit_count(const BitVec &bits) {
  int count = 0;
  for (uint64_t word : bits) {
    count += static_cast<int>(__builtin_popcountll(word));
  }
  return count;
}

bool is_empty(const BitVec &bits) {
  for (uint64_t word : bits) {
    if (word != 0ULL) {
      return false;
    }
  }
  return true;
}

bool bitvec_lex_less(const BitVec &a, const BitVec &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

size_t PairKeyHash::operator()(const PairKey &key) const noexcept {
  uint64_t a = static_cast<uint32_t>(key.first);
  uint64_t b = static_cast<uint32_t>(key.second);
  return static_cast<size_t>(wyhash64(a, b));
}

bool PairKeyEqual::operator()(const PairKey &a, const PairKey &b) const noexcept {
  return a.first == b.first && a.second == b.second;
}

