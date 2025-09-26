// Stochastic backtracking sampler with pybind11 bindings
// Maps closely to ALE-Rax UndatedDTL backtrace logic but consumes
// GPU-computed log Pi, log Pibar, log E, log Ebar, CCP splits and species helpers.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <random>
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

static constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

enum RecEventType {
  EVENT_S = 0,
  EVENT_SL = 1,
  EVENT_D = 2,
  EVENT_DL = 3,
  EVENT_T = 4,
  EVENT_TL = 5,
  EVENT_L = 6,     // unused here
  EVENT_Leaf = 7,
  EVENT_Invalid = 8
};

struct LocalEvent {
  RecEventType type;
  int cid;                 // parent clade id
  int species;             // species e
  int dest_species;        // for T/TL move; -1 if not applicable
  int left_cid;            // used for S/D/T orientation
  int right_cid;           // used for S/D/T orientation
  bool s_orientation_lr;   // true if S uses (left->leftChild, right->rightChild)
  bool t_dest_is_left;     // for T: true if left_cid goes to dest species
  double log_mass;
};

struct ScenarioEvent {
  RecEventType type;
  int gene_node;
  int species_node;
  int dest_species_node; // for transfers, else -1
  int left_gene_index;
  int right_gene_index;
  int cid;
  int left_cid;
  int right_cid;
};

struct CSRSegments {
  // For each parent clade segment in parents_sorted order, ptr gives [start,end) in sorted split arrays
  std::vector<int> ptr;               // size C+1
  std::vector<int> parents_sorted;    // size C
  std::vector<int> parent_to_seg;     // size C, maps parent id -> seg index [0..C-1] or -1 if none
};

static CSRSegments build_segments(const py::array_t<long long>& ptr_,
                                  const py::array_t<long long>& parents_sorted_,
                                  int C) {
  CSRSegments segs;
  segs.ptr.resize(C + 1);
  segs.parents_sorted.resize(C);
  segs.parent_to_seg.assign(C, -1);
  auto ptr = ptr_.unchecked<1>();
  auto ps  = parents_sorted_.unchecked<1>();
  if (ptr.shape(0) != C + 1 || ps.shape(0) != C) {
    throw std::runtime_error("Invalid ptr/parents_sorted sizes");
  }
  for (int i = 0; i < C + 1; ++i) segs.ptr[i] = static_cast<int>(ptr(i));
  for (int i = 0; i < C; ++i) {
    int p = static_cast<int>(ps(i));
    segs.parents_sorted[i] = p;
    if (p >= 0 && p < C && segs.parent_to_seg[p] == -1) {
      segs.parent_to_seg[p] = i;
    }
  }
  return segs;
}

static inline double logaddexp(double a, double b) {
  if (a == NEG_INF) return b;
  if (b == NEG_INF) return a;
  if (a < b) std::swap(a, b);
  return a + std::log1p(std::exp(b - a));
}

// Sample index from log-weights. Returns -1 if all -inf
static int sample_from_log_weights(const std::vector<double>& lw, std::mt19937_64& rng) {
  double m = NEG_INF;
  for (double v : lw) m = std::max(m, v);
  if (!std::isfinite(m)) return -1;
  double total = 0.0;
  std::vector<double> w(lw.size());
  for (size_t i = 0; i < lw.size(); ++i) {
    double wi = std::exp(lw[i] - m);
    w[i] = wi;
    total += wi;
  }
  if (total <= 0.0) return -1;
  std::uniform_real_distribution<double> U(0.0, 1.0);
  double r = U(rng) * total;
  double acc = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    acc += w[i];
    if (acc >= r) return static_cast<int>(i);
  }
  return static_cast<int>(w.size() - 1);
}

struct Inputs {
  int C;
  int S;
  // Matrices are row-major contiguous
  py::array_t<double> Pi_log;     // [C,S]
  py::array_t<double> Pibar_log;  // [C,S]
  py::array_t<double> E_log;      // [S]
  py::array_t<double> Ebar_log;   // [S]
  py::array_t<double> E_s1_log;   // [S]
  py::array_t<double> E_s2_log;   // [S]
  py::array_t<double> clade_species_log; // [C,S] -inf where unmapped

  // CCP splits sorted by parent (lefts/rights/logprobs aligned to sorted order)
  py::array_t<long long> split_lefts_sorted;   // [N_splits]
  py::array_t<long long> split_rights_sorted;  // [N_splits]
  py::array_t<double>    log_split_probs_sorted; // [N_splits]
  CSRSegments segs; // ptr[C+1], parents_sorted[C], parent_to_seg[C]

  // Species helpers
  std::vector<int> left_child;   // [S], -1 for leaves
  std::vector<int> right_child;  // [S]
  std::vector<char> species_is_leaf; // [S]

  // Clade helpers
  std::vector<char> clade_is_leaf;    // [C]

  // Recipients: dense row-stochastic matrix [S,S]
  py::array_t<double> Recipients; // [S,S]

  // Event log probabilities
  double log_pS;
  double log_pD;
  double log_pT;
  double log_2;
};

static inline double at2(const py::array_t<double>& A, int i, int j) {
  auto buf = A.unchecked<2>();
  return buf(i, j);
}

static inline double at1(const py::array_t<double>& A, int i) {
  auto buf = A.unchecked<1>();
  return buf(i);
}

static std::vector<LocalEvent> enumerate_events(const Inputs& in, int cid, int e) {
  std::vector<LocalEvent> events;
  events.reserve(64);
  bool sp_leaf = in.species_is_leaf[e];
  // Iterate CCP splits of this clade
  int seg = (cid >= 0 && cid < in.C) ? in.segs.parent_to_seg[cid] : -1;
  if (seg != -1) {
    int start = in.segs.ptr[seg];
    int end   = in.segs.ptr[seg + 1];
    for (int i = start; i < end; ++i) {
      int left  = static_cast<int>(in.split_lefts_sorted.unchecked<1>()(i));
      int right = static_cast<int>(in.split_rights_sorted.unchecked<1>()(i));
      double logf = in.log_split_probs_sorted.unchecked<1>()(i);
      // S events (only if species internal)
      if (!sp_leaf) {
        int f = in.left_child[e];
        int g = in.right_child[e];
        double mS1 = in.log_pS + logf + at2(in.Pi_log, left, f) + at2(in.Pi_log, right, g);
        double mS2 = in.log_pS + logf + at2(in.Pi_log, left, g) + at2(in.Pi_log, right, f);
        events.push_back({EVENT_S, cid, e, -1, left, right, true, false, mS1});
        events.push_back({EVENT_S, cid, e, -1, right, left, false, false, mS2});
      }
      // D event
      {
        double mD = in.log_pD + logf + at2(in.Pi_log, left, e) + at2(in.Pi_log, right, e);
        events.push_back({EVENT_D, cid, e, -1, left, right, true, false, mD});
      }
      // T events (two orientations)
      {
        double mT1 = in.log_pT + logf + at2(in.Pi_log, left, e) + at2(in.Pibar_log, right, e);
        double mT2 = in.log_pT + logf + at2(in.Pi_log, right, e) + at2(in.Pibar_log, left, e);
        events.push_back({EVENT_T, cid, e, -1, left, right, true,  false, mT1});  // right receives
        events.push_back({EVENT_T, cid, e, -1, left, right, false, true,  mT2});  // left receives
      }
    }
  }
  // SL (only if species internal)
  if (!sp_leaf) {
    int f = in.left_child[e];
    int g = in.right_child[e];
    double mSL1 = in.log_pS + at2(in.Pi_log, cid, f) + at1(in.E_s2_log, e); // uses E_s2(e)
    double mSL2 = in.log_pS + at2(in.Pi_log, cid, g) + at1(in.E_s1_log, e); // uses E_s1(e)
    events.push_back({EVENT_SL, cid, e, f, -1, -1, true, false, mSL1});
    events.push_back({EVENT_SL, cid, e, g, -1, -1, false, false, mSL2});
  }
  // DL
  {
    double mDL = in.log_2 + in.log_pD + at1(in.E_log, e) + at2(in.Pi_log, cid, e);
    events.push_back({EVENT_DL, cid, e, -1, -1, -1, false, false, mDL});
  }
  // TL variants
  {
    double mTL_dest_lost = in.log_pT + at2(in.Pi_log, cid, e) + at1(in.Ebar_log, e);
    double mTL_src_lost  = in.log_pT + at2(in.Pibar_log, cid, e) + at1(in.E_log, e);
    events.push_back({EVENT_TL, cid, e, -1, -1, -1, false, false, mTL_dest_lost}); // will resample
    events.push_back({EVENT_TL, cid, e, -1, -1, -1, false, true,  mTL_src_lost});  // move to dest
  }
  // Leaf
  if (in.clade_is_leaf[cid] && in.species_is_leaf[e]) {
    double leaf_mass = at2(in.clade_species_log, cid, e);
    if (std::isfinite(leaf_mass)) {
      double mLeaf = in.log_pS + leaf_mass;
      events.push_back({EVENT_Leaf, cid, e, -1, -1, -1, false, false, mLeaf});
    }
  }
  return events;
}

static int sample_transfer_destination(const Inputs& in, int origin_e, int cid_child, std::mt19937_64& rng) {
  // weights ∝ Recipients[origin_e, h] * exp(Pi[cid_child, h])
  auto Rec = in.Recipients.unchecked<2>();
  std::vector<double> lw(in.S, NEG_INF);
  for (int h = 0; h < in.S; ++h) {
    double r = Rec(origin_e, h);
    if (r <= 0.0) { lw[h] = NEG_INF; continue; }
    lw[h] = std::log(r) + at2(in.Pi_log, cid_child, h);
  }
  int idx = sample_from_log_weights(lw, rng);
  if (idx < 0) {
    // fallback: uniform over positive recipients
    std::vector<int> cand;
    for (int h = 0; h < in.S; ++h) if (Rec(origin_e, h) > 0.0) cand.push_back(h);
    if (cand.empty()) return origin_e; // degenerate
    std::uniform_int_distribution<int> U(0, static_cast<int>(cand.size()) - 1);
    return cand[U(rng)];
  }
  return idx;
}

struct Scenario {
  std::vector<ScenarioEvent> events;
};

struct SampleContext {
  std::unordered_map<uint64_t, int> state_visits; // key = (cid<<32)|e
  int max_depth = 100000;
  int max_events = 1000000;
  int max_state_visits = 10000;
  int event_count = 0;
};

static inline uint64_t make_key(int cid, int e) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(cid)) << 32) |
         static_cast<uint32_t>(e);
}

static void backtrack(const Inputs& in, int cid, int e, int gene_node,
                      std::mt19937_64& rng, Scenario& sc, int& gene_counter,
                      SampleContext& ctx, int depth = 0) {
  // Safety to avoid infinite loops in degenerate cases
  if (depth > ctx.max_depth) {
    throw std::runtime_error("Backtracking exceeded recursion limit; potential cycle");
  }
  if (++ctx.event_count > ctx.max_events) {
    throw std::runtime_error("Exceeded maximum allowed events during backtracking");
  }
  // Count visits to this (cid,e)
  auto key = make_key(cid, e);
  int visits = ++ctx.state_visits[key];
  // Enumerate, sample, and handle potential resampling events (DL and TL-dest-lost)
  LocalEvent ev;
  int resample_attempts = 0;
  while (true) {
    std::vector<LocalEvent> evs = enumerate_events(in, cid, e);
    if (evs.empty()) {
      throw std::runtime_error("No events available at state");
    }
    // If we visited this state too many times, suppress TL(source lost) to force progress
    bool suppress_tl_src_lost = (visits > ctx.max_state_visits);
    std::vector<double> lw;
    lw.reserve(evs.size());
    for (auto& evt : evs) {
      if (suppress_tl_src_lost && evt.type == EVENT_TL && evt.t_dest_is_left) {
        lw.push_back(NEG_INF);
      } else {
        lw.push_back(evt.log_mass);
      }
    }
    int choice = sample_from_log_weights(lw, rng);
    if (choice < 0) {
      throw std::runtime_error("All event masses are -inf at state");
    }
    ev = evs[choice];
    // Resample-only events: DL, TL with dest lost (we encoded dest-lost with t_dest_is_left=false)
    bool is_resample_only = (ev.type == EVENT_DL) || (ev.type == EVENT_TL && !ev.t_dest_is_left);
    if (!is_resample_only) {
      break; // process event
    }
    // protect against pathological resampling loops
    if (++resample_attempts > 100000) {
      throw std::runtime_error("Exceeded resampling attempts at a state (likely degenerate probabilities)");
    }
    // loop and resample at same (cid,e)
  }

  // Handle according to type
  switch (ev.type) {
    case EVENT_S: {
      // Generate two gene children
      int left_gene = gene_counter++;
      int right_gene = gene_counter++;
      sc.events.push_back({EVENT_S, gene_node, e, -1, left_gene, right_gene, cid, ev.left_cid, ev.right_cid});
      int f = in.left_child[e];
      int g = in.right_child[e];
      // Orientation already encoded in left_cid/right_cid
      backtrack(in, ev.left_cid, f, left_gene, rng, sc, gene_counter, ctx, depth + 1);
      backtrack(in, ev.right_cid, g, right_gene, rng, sc, gene_counter, ctx, depth + 1);
      break;
    }
    case EVENT_D: {
      int left_gene = gene_counter++;
      int right_gene = gene_counter++;
      sc.events.push_back({EVENT_D, gene_node, e, -1, left_gene, right_gene, cid, ev.left_cid, ev.right_cid});
      backtrack(in, ev.left_cid, e, left_gene, rng, sc, gene_counter, ctx, depth + 1);
      backtrack(in, ev.right_cid, e, right_gene, rng, sc, gene_counter, ctx, depth + 1);
      break;
    }
    case EVENT_T: {
      int left_gene = gene_counter++;
      int right_gene = gene_counter++;
      // Decide which child goes to dest species
      int cid_src, cid_dest;
      if (ev.t_dest_is_left) {
        cid_src  = ev.right_cid;
        cid_dest = ev.left_cid;
      } else {
        cid_src  = ev.left_cid;
        cid_dest = ev.right_cid;
      }
      int dest = sample_transfer_destination(in, e, cid_dest, rng);
      sc.events.push_back({EVENT_T, gene_node, e, dest, left_gene, right_gene, cid, ev.left_cid, ev.right_cid});
      // Recurse: source branch stays in e, dest branch goes to dest
      if (ev.t_dest_is_left) {
        backtrack(in, ev.right_cid, e, left_gene, rng, sc, gene_counter, ctx, depth + 1);
        backtrack(in, ev.left_cid, dest, right_gene, rng, sc, gene_counter, ctx, depth + 1);
      } else {
        backtrack(in, ev.left_cid, e, left_gene, rng, sc, gene_counter, ctx, depth + 1);
        backtrack(in, ev.right_cid, dest, right_gene, rng, sc, gene_counter, ctx, depth + 1);
      }
      break;
    }
    case EVENT_SL: {
      // No new gene children; move to surviving child species
      int dest_child = ev.dest_species; // either left or right child id
      sc.events.push_back({EVENT_SL, gene_node, e, dest_child, -1, -1, cid, -1, -1});
      backtrack(in, cid, dest_child, gene_node, rng, sc, gene_counter, ctx, depth + 1);
      break;
    }
    case EVENT_DL: {
      // Should not reach here due to resample loop above
      break;
    }
    case EVENT_TL: {
      if (ev.t_dest_is_left) {
        // Source lost variant selected (we encoded via t_dest_is_left=true): move lineage to some dest and continue
        int dest = sample_transfer_destination(in, e, cid, rng);
        sc.events.push_back({EVENT_TL, gene_node, e, dest, -1, -1, cid, -1, -1});
        backtrack(in, cid, dest, gene_node, rng, sc, gene_counter, ctx, depth + 1);
      } else {
        // Dest lost: should not reach here due to resample loop above
      }
      break;
    }
    case EVENT_Leaf: {
      sc.events.push_back({EVENT_Leaf, gene_node, e, -1, -1, -1, cid, -1, -1});
      break;
    }
    default:
      throw std::runtime_error("Invalid event type sampled");
  }
}

py::list sample_scenarios(
    py::array_t<double> Pi_log,
    py::array_t<double> Pibar_log,
    py::array_t<double> E_log,
    py::array_t<double> Ebar_log,
    py::array_t<double> E_s1_log,
    py::array_t<double> E_s2_log,
    py::array_t<long long> split_lefts_sorted,
    py::array_t<long long> split_rights_sorted,
    py::array_t<double> log_split_probs_sorted,
    py::array_t<long long> ptr,
    py::array_t<long long> parents_sorted,
    py::array_t<long long> left_child,
    py::array_t<long long> right_child,
    py::array_t<char> species_is_leaf,
    py::array_t<char> clade_is_leaf,
    py::array_t<double> Recipients,
    py::array_t<double> clade_species_log,
    double log_pS,
    double log_pD,
    double log_pT,
    double log_2,
    int root_cid,
    int samples,
    unsigned long long seed
) {
  // Validate shapes
  if (Pi_log.ndim() != 2 || Pibar_log.ndim() != 2) throw std::runtime_error("Pi_log/Pibar_log must be 2D");
  if (E_log.ndim() != 1 || Ebar_log.ndim() != 1) throw std::runtime_error("E arrays must be 1D");
  if (Recipients.ndim() != 2) throw std::runtime_error("Recipients must be 2D");
  int C = static_cast<int>(Pi_log.shape(0));
  int S = static_cast<int>(Pi_log.shape(1));
  if (Pibar_log.shape(0) != C || Pibar_log.shape(1) != S) throw std::runtime_error("Pibar shape mismatch");
  if (clade_species_log.shape(0) != C || clade_species_log.shape(1) != S) throw std::runtime_error("clade_species_log shape mismatch");
  if (E_log.shape(0) != S || Ebar_log.shape(0) != S) throw std::runtime_error("E shapes mismatch S");
  if (Recipients.shape(0) != S || Recipients.shape(1) != S) throw std::runtime_error("Recipients shape mismatch SxS");
  if (left_child.shape(0) != S || right_child.shape(0) != S) throw std::runtime_error("child arrays size S");
  if (species_is_leaf.shape(0) != S) throw std::runtime_error("species_is_leaf size S");
  if (clade_is_leaf.shape(0) != C) throw std::runtime_error("clade_is_leaf size C");

  Inputs in;
  in.C = C; in.S = S;
  in.Pi_log = Pi_log;
  in.Pibar_log = Pibar_log;
  in.E_log = E_log;
  in.Ebar_log = Ebar_log;
  in.E_s1_log = E_s1_log;
  in.E_s2_log = E_s2_log;
  in.clade_species_log = clade_species_log;
  in.split_lefts_sorted = split_lefts_sorted;
  in.split_rights_sorted = split_rights_sorted;
  in.log_split_probs_sorted = log_split_probs_sorted;
  in.segs = build_segments(ptr, parents_sorted, C);
  in.left_child.resize(S);
  in.right_child.resize(S);
  in.species_is_leaf.resize(S);
  {
    auto lc = left_child.unchecked<1>();
    auto rc = right_child.unchecked<1>();
    auto sl = species_is_leaf.unchecked<1>();
    for (int e = 0; e < S; ++e) {
      in.left_child[e] = static_cast<int>(lc(e));
      in.right_child[e] = static_cast<int>(rc(e));
      in.species_is_leaf[e] = sl(e);
    }
  }
  in.clade_is_leaf.resize(C);
  {
    auto cl = clade_is_leaf.unchecked<1>();
    for (int c = 0; c < C; ++c) in.clade_is_leaf[c] = cl(c);
  }
  in.Recipients = Recipients;
  in.log_pS = log_pS;
  in.log_pD = log_pD;
  in.log_pT = log_pT;
  in.log_2  = log_2;

  std::mt19937_64 rng(seed);
  py::list out;
  for (int i = 0; i < samples; ++i) {
    Scenario sc;
    int gene_root = 0; // virtual root id
    int gene_counter = 1;
    SampleContext ctx;
    // Sample origin species proportional to Pi[root_cid,:]
    std::vector<double> lw_origin(S, NEG_INF);
    for (int e = 0; e < S; ++e) lw_origin[e] = at2(in.Pi_log, root_cid, e);
    int origin_e = sample_from_log_weights(lw_origin, rng);
    if (origin_e < 0) origin_e = 0;
    backtrack(in, root_cid, origin_e, gene_root, rng, sc, gene_counter, ctx, 0);
    // Convert to Python dict
    py::list evlist;
    for (const auto& ev : sc.events) {
      py::dict d;
      d["type"] = static_cast<int>(ev.type);
      d["gene_node"] = ev.gene_node;
      d["species_node"] = ev.species_node;
      d["dest_species_node"] = ev.dest_species_node;
      d["left_gene_index"] = ev.left_gene_index;
      d["right_gene_index"] = ev.right_gene_index;
      d["cid"] = ev.cid;
      d["left_cid"] = ev.left_cid;
      d["right_cid"] = ev.right_cid;
      evlist.append(d);
    }
    py::dict scd;
    scd["events"] = evlist;
    scd["origin_species"] = origin_e;
    scd["gene_root"] = gene_root;
    out.append(scd);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ALE-style stochastic backtracking sampler (pybind11)";
  m.def("sample_scenarios", &sample_scenarios,
        py::arg("Pi_log"), py::arg("Pibar_log"), py::arg("E_log"), py::arg("Ebar_log"),
        py::arg("E_s1_log"), py::arg("E_s2_log"),
        py::arg("split_lefts_sorted"), py::arg("split_rights_sorted"), py::arg("log_split_probs_sorted"),
        py::arg("ptr"), py::arg("parents_sorted"),
        py::arg("left_child"), py::arg("right_child"),
        py::arg("species_is_leaf"), py::arg("clade_is_leaf"),
        py::arg("Recipients"), py::arg("clade_species_log"),
        py::arg("log_pS"), py::arg("log_pD"), py::arg("log_pT"), py::arg("log_2"),
        py::arg("root_cid"), py::arg("samples"), py::arg("seed")
  );
}
