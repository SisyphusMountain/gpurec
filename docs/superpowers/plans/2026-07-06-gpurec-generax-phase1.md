# gpurec-backed GeneRax — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GeneRax-equivalent tool (`gpurax`) that infers ML reconciled gene-family trees under UndatedDTL by maximizing the per-family joint likelihood `L(S,N|G)·L(G|A)`, using **gpurec** (GPU, batched) for the reconciliation term and **libpll/coraxlib** (CPU, via a pybind11 wrapper reusing GeneRax's `LibpllEvaluation`) for the sequence term.

**Architecture:** A Python orchestration owns the search loop; a thin C++ pybind11 extension (`gpurax._seqlik`) exposes GeneRax's sequence-likelihood + SPR primitives per family. Each SPR step enumerates neighbor topologies for all active families, scores the sequence term on a CPU thread pool and the reconciliation term for the entire (family × neighbor) set in **one batched gpurec call** (each candidate topology is a pseudo-family in genewise mode with fixed rates), then applies the best joint-improving move per family. DTL rates are optimized between radii with gpurec's optimizers.

**Tech Stack:** Python 3, PyTorch + Triton (gpurec), pybind11 + CMake, coraxlib/GeneRaxCore C++, ETE3/dendropy for tree utilities (RF), pytest.

## Global Constraints

- gpurec works entirely in **log2 / bits**; rates live in log2 space; theta order is **`[log2 D, log2 L, log2 T]`** (index 2 = transfer). Convert to natural-log for GeneRax comparison: `logL_nats = -nll_bits * math.log(2)`. (verbatim: `bits_to_nats(x) = x * math.log(2.0)`, `gpurec/cli/_common.py:8`)
- gpurec `gene_trees` are **file paths only** — no in-memory Newick string path through the constructor (`gpurec/api/model.py:54`, `crates/gpurec-preprocess/src/lib.rs:154`). In-memory topologies must be written to files (or fed through `build_family_ccp`).
- **Never use `clamp`/`clamp_`** on any values (project rule — corrupts results silently).
- Import the **repo** gpurec, not the stale `.venv` build: insert repo root on `sys.path` before `import gpurec` (project rule).
- Reconciliation term is **UndatedDTL** only in Phase 1 (UndatedDL if trivially available).
- Single-node only: **GPU + CPU thread pool**, no MPI.
- Do not reintroduce EB priors or `clamp` into the rate optimizer; the regularizer is Sanderson penalty + CV (project rule).
- Package lives at repo root as a sibling of `gpurec/`: `gpurax/`. Tests under `tests/gpurax/`. Fixtures under `tests/gpurax/fixtures/`.

## File Structure

```
gpurax/
  __init__.py                # public API re-exports
  _seqlik/                   # C++ pybind11 extension
    CMakeLists.txt
    seqlik.cpp               # SeqFamily wrapper over LibpllEvaluation + SPR
  io/
    families.py              # GeneRax families-file + species-tree + mapping parsing
    initial_trees.py         # Step 0: build starting trees when absent
  recon/
    adapter.py               # gpurec batch reconciliation over candidate topologies
    parity.py                # single fixed-tree recon logL (validation primitive)
  search/
    scorer.py                # JointScorer: seq (threaded) + recWeight*rec (gpurec)
    spr.py                   # batch-native ragged multi-family SPR loop
  rates/
    optimize.py              # gpurec DTL-rate optimization adapter
  reconcile/
    sample.py                # final reconciliation via gpurec backtracking
    export.py                # Newick / scores.tsv / RecPhyloXML / Notung
  driver.py                  # orchestration: Step0 -> warm-up -> (rates<->SPR) -> reconcile
  cli.py                     # argparse CLI mirroring GeneRax
tests/gpurax/
  fixtures/                  # tiny 4-taxon dataset + captured GeneRax reference outputs
  ...                        # one test module per task
```

---

## Milestone A — Environment, references, build foundation

### Task A1: Package scaffold + tiny fixture + captured GeneRax reference

**Files:**
- Create: `gpurax/__init__.py`
- Create: `tests/gpurax/fixtures/species.nwk`, `gene.nwk`, `aln.fasta`, `map.link`
- Create: `tests/gpurax/fixtures/generax_ref.json` (captured)
- Create: `tests/gpurax/test_scaffold.py`
- Create: `tests/gpurax/conftest.py`

**Interfaces:**
- Produces: `gpurax.__version__` (str); pytest fixtures `fixture_dir` (Path), `generax_ref` (dict) available to all later tests.

- [ ] **Step 1: Write the 4-taxon fixture files**

`species.nwk`:
```
((A,B),(C,D));
```
`gene.nwk` (one gene per species; leaf names carry species suffix):
```
((a_A:0.1,b_B:0.1):0.1,(c_C:0.1,d_D:0.1):0.1);
```
`aln.fasta` (4 short DNA sequences, deliberately low-signal):
```
>a_A
ACGTACGTACGTAAGT
>b_B
ACGTACGTACGTAAGT
>c_C
ACGTTCGTACGTTAGT
>d_D
ACGTTCGTACGTTAGT
```
`map.link` (GeneRax mapping, one `gene species` per line):
```
a_A A
b_B B
c_C C
d_D D
```

- [ ] **Step 2: Locate or build a GeneRax binary and capture reference outputs**

Run (discover an existing build first, then fall back to building):
```bash
find /home/enzo/Documents/git/gpurec/consolidate-release -name generax -type f -perm -u+x 2>/dev/null | head
```
Expected: a path such as `.../experiments/ghost_lineages/tools/GeneRax/build/bin/generax`. If none, build per `generax_gpurec/generax/README` (`./install.sh` or CMake). Then create a GeneRax families file `tests/gpurax/fixtures/families.txt`:
```
[FAMILIES]
- fam1
starting_gene_tree = gene.nwk
alignment = aln.fasta
mapping = map.link
subst_model = GTR
```
Run GeneRax in EVAL mode to capture the joint/rec/seq breakdown at fixed rates, and in full SPR mode to capture the reconciled tree:
```bash
generax -f tests/gpurax/fixtures/families.txt -s tests/gpurax/fixtures/species.nwk \
  -r UndatedDTL --strategy EVAL -p /tmp/gx_eval \
  --dup-rate 0.2 --loss-rate 0.3 --transfer-rate 0.1
```
Capture from `/tmp/gx_eval` logs/outputs into `generax_ref.json`:
```json
{
  "fixed_rates": {"D": 0.2, "L": 0.3, "T": 0.1},
  "rec_loglk_nats": <captured>,
  "seq_loglk_nats": <captured>,
  "joint_loglk_nats": <captured>,
  "reconciled_newick": "<captured from SPR run>"
}
```
If GeneRax cannot be built in this environment, set `generax_ref.json` to `{"available": false}` and mark all parity-vs-GeneRax steps `xfail(reason="no GeneRax binary")` — the seq/recon *self-consistency* tests (B1 internal, C2) still gate. Record which path was taken in the commit message.

- [ ] **Step 3: Write `gpurax/__init__.py` and scaffold test**

```python
# gpurax/__init__.py
__version__ = "0.0.1"
```
```python
# tests/gpurax/conftest.py
import json, pathlib, pytest
@pytest.fixture
def fixture_dir():
    return pathlib.Path(__file__).parent / "fixtures"
@pytest.fixture
def generax_ref(fixture_dir):
    return json.loads((fixture_dir / "generax_ref.json").read_text())
```
```python
# tests/gpurax/test_scaffold.py
import gpurax
def test_version():
    assert gpurax.__version__
def test_fixtures_exist(fixture_dir):
    for f in ["species.nwk", "gene.nwk", "aln.fasta", "map.link"]:
        assert (fixture_dir / f).exists()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/gpurax/test_scaffold.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add gpurax/__init__.py tests/gpurax/
git commit -m "feat(gpurax): package scaffold, 4-taxon fixture, GeneRax reference capture"
```

### Task A2: pybind11 + CMake build linking coraxlib/GeneRaxCore

**Files:**
- Create: `gpurax/_seqlik/CMakeLists.txt`
- Create: `gpurax/_seqlik/seqlik.cpp` (minimal)
- Create: `pyproject.toml` (or extend existing) with scikit-build-core backend
- Create: `tests/gpurax/test_build.py`

**Interfaces:**
- Produces: importable `gpurax._seqlik` with `gpurax._seqlik.corax_version() -> str`.

- [ ] **Step 1: Write the failing import test**

```python
# tests/gpurax/test_build.py
def test_extension_imports():
    from gpurax import _seqlik
    assert isinstance(_seqlik.corax_version(), str)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: gpurax._seqlik`.

- [ ] **Step 3: Determine the minimal GeneRaxCore/coraxlib sources to compile**

Run to inventory what `LibpllEvaluation` transitively needs:
```bash
cd generax_gpurec/generax/ext/GeneRaxCore
ls src/likelihoods/LibpllEvaluation.cpp src/trees/PLLUnrootedTree.cpp \
   src/trees/PLLTreeInfo.cpp src/IO/LibpllParsers.cpp src/IO/Model.cpp 2>&1
find ext/coraxlib -name 'libcorax*' -o -name 'CMakeLists.txt' | head
```
Expected: the `.cpp` files exist; coraxlib provides its own CMake target `corax`. Note their paths for the CMakeLists `add_library` source list.

- [ ] **Step 4: Write CMakeLists.txt and minimal seqlik.cpp**

```cmake
# gpurax/_seqlik/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(seqlik LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(GRC ${CMAKE_SOURCE_DIR}/../../generax_gpurec/generax/ext/GeneRaxCore)
add_subdirectory(${GRC}/ext/coraxlib corax_build)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_seqlik
    seqlik.cpp
    ${GRC}/src/likelihoods/LibpllEvaluation.cpp
    ${GRC}/src/trees/PLLUnrootedTree.cpp
    ${GRC}/src/trees/PLLTreeInfo.cpp
    ${GRC}/src/IO/LibpllParsers.cpp
    ${GRC}/src/IO/Model.cpp)
target_include_directories(_seqlik PRIVATE ${GRC}/src ${GRC}/ext/coraxlib/src)
target_link_libraries(_seqlik PRIVATE corax)
```
```cpp
// gpurax/_seqlik/seqlik.cpp
#include <pybind11/pybind11.h>
#include <string>
namespace py = pybind11;
static std::string corax_version() { return "corax-linked"; }
PYBIND11_MODULE(_seqlik, m) {
    m.def("corax_version", &corax_version);
}
```
Add build backend to `pyproject.toml` (scikit-build-core) so `pip install -e .` compiles the extension into `gpurax/_seqlik`. If the transitive source list from Step 3 differs, extend `pybind11_add_module` accordingly (compile errors name the missing symbols → add the defining `.cpp`).

- [ ] **Step 5: Build and run**

Run:
```bash
pip install -e . --no-build-isolation 2>&1 | tail -20
pytest tests/gpurax/test_build.py -v
```
Expected: build succeeds; test PASS.

- [ ] **Step 6: Commit**

```bash
git add gpurax/_seqlik/ pyproject.toml tests/gpurax/test_build.py
git commit -m "build(gpurax): pybind11+CMake _seqlik extension linking coraxlib/GeneRaxCore"
```

---

## Milestone B — Sequence-likelihood wrapper (`_seqlik`)

### Task B1: `SeqFamily` — load + sequence log-likelihood

**Files:**
- Modify: `gpurax/_seqlik/seqlik.cpp`
- Create: `tests/gpurax/test_seqlik_loglk.py`

**Interfaces:**
- Consumes: `LibpllEvaluation(const std::string&, bool isFile, const std::string& aln, const std::string& model)` (`LibpllEvaluation.hpp:43`); `double computeLikelihood(bool incremental)` (`:54`); `double optimizeBranches(double)` (`:61`); `PLLUnrootedTree& getGeneTree()` (`:89`); `std::string PLLUnrootedTree::getNewickString(...)` (`PLLUnrootedTree.hpp:129`).
- Produces (Python): class `SeqFamily(newick_str: str, alignment_path: str, model: str)` with methods `seq_loglk(opt_bl: bool=False) -> float` (natural-log), `newick() -> str`.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/gpurax/test_seqlik_loglk.py
import math, pytest
from gpurax._seqlik import SeqFamily

def test_seq_loglk_matches_generax(fixture_dir, generax_ref):
    if not generax_ref.get("available", True):
        pytest.xfail("no GeneRax binary")
    newick = (fixture_dir / "gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(fixture_dir / "aln.fasta"), "GTR")
    ll = fam.seq_loglk(opt_bl=False)   # nats, no BL optimization
    assert ll == pytest.approx(generax_ref["seq_loglk_nats"], rel=1e-6)

def test_seq_loglk_is_finite_and_negative():
    # self-consistency, always runs
    newick = "((a_A:0.1,b_B:0.1):0.1,(c_C:0.1,d_D:0.1):0.1);"
    import pathlib
    fx = pathlib.Path(__file__).parent / "fixtures"
    fam = SeqFamily(newick, str(fx / "aln.fasta"), "GTR")
    ll = fam.seq_loglk(opt_bl=False)
    assert math.isfinite(ll) and ll < 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_seqlik_loglk.py -v`
Expected: FAIL — `SeqFamily` not defined.

- [ ] **Step 3: Implement `SeqFamily` in seqlik.cpp**

```cpp
#include <pybind11/pybind11.h>
#include <memory>
#include <string>
#include "likelihoods/LibpllEvaluation.hpp"
#include "trees/PLLUnrootedTree.hpp"
namespace py = pybind11;

class SeqFamily {
public:
  SeqFamily(const std::string &newick, const std::string &aln, const std::string &model)
    : _eval(newick, /*isNewickAFile=*/false, aln, model) {}
  double seq_loglk(bool opt_bl) {
    if (opt_bl) _eval.optimizeBranches();
    return _eval.computeLikelihood(/*incremental=*/false);
  }
  std::string newick() { return _eval.getGeneTree().getNewickString(); }
private:
  LibpllEvaluation _eval;
};

PYBIND11_MODULE(_seqlik, m) {
  m.def("corax_version", []{ return std::string("corax-linked"); });
  py::class_<SeqFamily>(m, "SeqFamily")
    .def(py::init<const std::string&, const std::string&, const std::string&>(),
         py::arg("newick"), py::arg("alignment_path"), py::arg("model"))
    .def("seq_loglk", &SeqFamily::seq_loglk, py::arg("opt_bl") = false)
    .def("newick", &SeqFamily::newick);
}
```

- [ ] **Step 4: Rebuild and run**

Run:
```bash
pip install -e . --no-build-isolation 2>&1 | tail -5
pytest tests/gpurax/test_seqlik_loglk.py -v
```
Expected: `test_seq_loglk_is_finite_and_negative` PASS; parity test PASS (or xfail if no GeneRax). If parity is off, first check the model string (GeneRax default may be `GTR+G` — align the fixture's `subst_model` in A1 Step 2 with the string passed here).

- [ ] **Step 5: Commit**

```bash
git add gpurax/_seqlik/seqlik.cpp tests/gpurax/test_seqlik_loglk.py
git commit -m "feat(seqlik): SeqFamily sequence log-likelihood wrapper over LibpllEvaluation"
```

### Task B2: SPR neighbor enumeration + apply/rollback

**Files:**
- Modify: `gpurax/_seqlik/seqlik.cpp`
- Create: `tests/gpurax/test_seqlik_spr.py`

**Interfaces:**
- Consumes: `corax_utree_spr(corax_unode_t*, corax_unode_t*, corax_tree_rollback_t*)` (`utree_moves.h:107`); `corax_tree_rollback(...)` (`:144`); `PLLUnrootedTree::getNode(unsigned)` (`PLLUnrootedTree.hpp:115`), `getBranches()` (`:186`); the neighbor recursion pattern from `SearchUtils.cpp:diggRecursive` (regraft over `node->next->back` / `node->next->next->back`, pushing `node_index` along the path).
- Produces (Python): on `SeqFamily`: `spr_neighbors(radius: int) -> list[tuple[int,int,list[int]]]` returning `(prune_index, regraft_index, path)` moves; `apply_spr(prune_index: int, regraft_index: int) -> None`; `rollback() -> None`; `tree_hash() -> int`.

- [ ] **Step 1: Write the failing test (enumerate + apply/rollback round-trip)**

```python
# tests/gpurax/test_seqlik_spr.py
import pathlib
from gpurax._seqlik import SeqFamily
FX = pathlib.Path(__file__).parent / "fixtures"

def make():
    newick = (FX / "gene.nwk").read_text().strip()
    return SeqFamily(newick, str(FX / "aln.fasta"), "GTR")

def test_neighbors_nonempty_at_radius1():
    fam = make()
    nbrs = fam.spr_neighbors(1)
    assert len(nbrs) > 0
    p, r, path = nbrs[0]
    assert isinstance(p, int) and isinstance(r, int) and isinstance(path, list)

def test_apply_then_rollback_restores_tree():
    fam = make()
    h0 = fam.tree_hash()
    p, r, _ = fam.spr_neighbors(1)[0]
    fam.apply_spr(p, r)
    assert fam.tree_hash() != h0
    fam.rollback()
    assert fam.tree_hash() == h0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_seqlik_spr.py -v`
Expected: FAIL — `spr_neighbors` not defined.

- [ ] **Step 3: Implement enumeration + apply/rollback**

Add to `SeqFamily` (uses coraxlib nodes; enumeration mirrors `diggRecursive` but only records moves, no scoring). Include `<vector>`, `<unordered_set>`, and the coraxlib move header (`corax/tree/utree_moves.h` via `LibpllParsers` includes). Store a `corax_tree_rollback_t _lastRollback;` for `rollback()`.
```cpp
  std::vector<std::tuple<unsigned,unsigned,std::vector<unsigned>>> spr_neighbors(unsigned radius) {
    std::vector<std::tuple<unsigned,unsigned,std::vector<unsigned>>> out;
    auto &tree = _eval.getGeneTree();
    for (auto *prune : tree.getBranches()) {
      // seed from prune->next->back and prune->next->next->back, recurse to `radius`
      std::vector<unsigned> path;
      collect(prune, prune->next->back, path, 1, radius, out, prune->node_index);
      collect(prune, prune->next->next->back, path, 1, radius, out, prune->node_index);
    }
    return out;
  }
  void apply_spr(unsigned p, unsigned r) {
    auto &tree = _eval.getGeneTree();
    corax_unode_t *pe = tree.getNode(p), *re = tree.getNode(r);
    int rc = corax_utree_spr(pe, re, &_lastRollback);
    if (rc != CORAX_SUCCESS) throw std::runtime_error("corax_utree_spr failed");
  }
  void rollback() { corax_tree_rollback(&_lastRollback); }
  size_t tree_hash() { return _eval.getGeneTree().getUnrootedTreeHash(); }
```
where `collect(prune, regraft, path, radius, maxRadius, out, pruneIdx)` is a private recursive helper that (a) if the (prune,regraft) pair is a valid SPR move — reuse the validity check `sprYeldsSameTree` logic — appends `(pruneIdx, regraft->node_index, path)` to `out`, then (b) if `radius < maxRadius`, pushes `regraft->node_index`, recurses on `regraft->next->back` and `regraft->next->next->back` at `radius+1`, and pops. Bind the three methods with `.def(...)` in the module block. Copy the `sprYeldsSameTree` predicate from `SPRSearch.cpp:47` to skip no-op moves.

- [ ] **Step 4: Rebuild and run**

Run:
```bash
pip install -e . --no-build-isolation 2>&1 | tail -5
pytest tests/gpurax/test_seqlik_spr.py -v
```
Expected: PASS (2 tests). If `tree_hash` after rollback differs, verify branch lengths are restored by the rollback (coraxlib's SPR rollback restores `prune_bl`/`regraft_bl`); the unrooted-topology hash should match regardless of BL.

- [ ] **Step 5: Commit**

```bash
git add gpurax/_seqlik/seqlik.cpp tests/gpurax/test_seqlik_spr.py
git commit -m "feat(seqlik): SPR neighbor enumeration + apply/rollback"
```

---

## Milestone C — Reconciliation batch adapter (gpurec)

### Task C1: Single fixed-tree reconciliation logL (parity primitive)

**Files:**
- Create: `gpurax/recon/__init__.py`, `gpurax/recon/parity.py`
- Create: `tests/gpurax/test_recon_parity.py`

**Interfaces:**
- Consumes: `GeneReconModel(species_tree, gene_trees, *, mode, device, dtype)` (`gpurec/api/model.py:33`); `model()` scalar NLL bits (`:420`); `bits_to_nats` (`gpurec/cli/_common.py:8`). Rate-set pattern from `gpurec/cli/reconcile.py:18-27`.
- Produces: `recon_loglk(species_path: str, gene_tree_path: str, D: float, L: float, T: float, *, device="cuda") -> float` (natural-log reconciliation logL for the single fixed tree).

- [ ] **Step 1: Write the failing parity test**

```python
# tests/gpurax/test_recon_parity.py
import pytest
from gpurax.recon.parity import recon_loglk

def test_recon_matches_generax(fixture_dir, generax_ref):
    if not generax_ref.get("available", True):
        pytest.xfail("no GeneRax binary")
    r = generax_ref["fixed_rates"]
    ll = recon_loglk(str(fixture_dir / "species.nwk"), str(fixture_dir / "gene.nwk"),
                     r["D"], r["L"], r["T"], device="cpu")
    assert ll == pytest.approx(generax_ref["rec_loglk_nats"], rel=1e-4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_recon_parity.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `recon_loglk`**

```python
# gpurax/recon/parity.py
import math, sys, pathlib, torch
_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)          # repo gpurec, not .venv (project rule)
from gpurec.api.model import GeneReconModel

def recon_loglk(species_path, gene_tree_path, D, L, T, *, device="cuda"):
    model = GeneReconModel(species_path, [gene_tree_path], mode="global",
                           device=device, dtype=torch.float64)
    triple = torch.tensor([math.log2(D), math.log2(L), math.log2(T)],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        model.theta.copy_(triple)
        nll_bits = float(model())          # summed scalar NLL in bits
    return -nll_bits * math.log(2.0)       # -> nats logL
```

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_recon_parity.py -v`
Expected: PASS (or xfail). **If parity fails, this is the rooting-convention risk (spec §8):** capture the discrepancy and proceed to Task C4 before continuing — do not paper over it.

- [ ] **Step 5: Commit**

```bash
git add gpurax/recon/ tests/gpurax/test_recon_parity.py
git commit -m "feat(recon): single fixed-tree reconciliation logL parity primitive"
```

### Task C2: Batched reconciliation over candidate topologies (pseudo-families)

**Files:**
- Create: `gpurax/recon/adapter.py`
- Create: `tests/gpurax/test_recon_batch.py`

**Interfaces:**
- Consumes: `GeneReconModel(..., mode="genewise")`; `model.genewise_loss_vector() -> Tensor[F]` NLL bits (`gpurec/api/model.py:235`); theta shape `(F,3)` in genewise mode (`gpurec/api/model.py:70`).
- Produces: class `ReconBatch(species_path: str, device="cuda")` with method `score(topology_paths: list[str], rates: list[tuple[float,float,float]]) -> list[float]` returning per-topology reconciliation logL (nats). `len(rates) == len(topology_paths)` (one rate triple per candidate; broadcast a single family's rate across its neighbors at the call site).

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_recon_batch.py
import pytest
from gpurax.recon.adapter import ReconBatch
from gpurax.recon.parity import recon_loglk

def test_batch_of_duplicates_is_uniform(fixture_dir):
    rb = ReconBatch(str(fixture_dir / "species.nwk"), device="cpu")
    paths = [str(fixture_dir / "gene.nwk")] * 3
    rates = [(0.2, 0.3, 0.1)] * 3
    lls = rb.score(paths, rates)
    assert len(lls) == 3
    assert lls[0] == pytest.approx(lls[1]) == pytest.approx(lls[2])

def test_batch_matches_single(fixture_dir):
    rb = ReconBatch(str(fixture_dir / "species.nwk"), device="cpu")
    ll_batch = rb.score([str(fixture_dir / "gene.nwk")], [(0.2, 0.3, 0.1)])[0]
    ll_single = recon_loglk(str(fixture_dir / "species.nwk"),
                            str(fixture_dir / "gene.nwk"), 0.2, 0.3, 0.1, device="cpu")
    assert ll_batch == pytest.approx(ll_single, rel=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_recon_batch.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `ReconBatch`**

Each candidate topology becomes one pseudo-family; genewise mode gives a per-candidate NLL vector. Set each theta row to that candidate's fixed rates.
```python
# gpurax/recon/adapter.py
import math, sys, pathlib, torch
_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from gpurec.api.model import GeneReconModel

class ReconBatch:
    def __init__(self, species_path, device="cuda"):
        self._species = species_path
        self._device = device

    def score(self, topology_paths, rates):
        model = GeneReconModel(self._species, list(topology_paths),
                               mode="genewise", device=self._device, dtype=torch.float64)
        theta = torch.empty_like(model.theta)          # (F, 3)
        for i, (D, L, T) in enumerate(rates):
            theta[i, 0] = math.log2(D); theta[i, 1] = math.log2(L); theta[i, 2] = math.log2(T)
        with torch.no_grad():
            model.theta.copy_(theta)
            nll_bits = model.genewise_loss_vector()     # [F] NLL bits
        return [(-float(b) * math.log(2.0)) for b in nll_bits]
```

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_recon_batch.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add gpurax/recon/adapter.py tests/gpurax/test_recon_batch.py
git commit -m "feat(recon): batched reconciliation over candidate topologies (genewise pseudo-families)"
```

### Task C3: Single-topology preprocess-cost probe (spec §7 risk #1)

**Files:**
- Create: `gpurax/recon/incremental.py`
- Create: `tests/gpurax/test_recon_cost.py`
- Create: `docs/gpurax/recon_cost_findings.md`

**Interfaces:**
- Consumes: `build_family_ccp(gene_path)` + `FamilyCcp.prune(...)` and `plan_batch_layouts(families_json, ...)` (`crates/gpurec-preprocess/src/lib.rs:2263, 2043, 124`); Python wrappers in `gpurec/core/scheduling/batching.py` (`preprocess_dataset:32`, `plan_batch_wave_layouts:139`); `model.replan_batches(...)` (`gpurec/api/model.py:250`).
- Produces: `preprocess_families(species_path, topology_paths) -> families_json` (per-topology dicts assembled WITHOUT a full-dataset re-parse of unchanged families); a recorded throughput number in `recon_cost_findings.md`.

- [ ] **Step 1: Write the throughput + correctness test**

```python
# tests/gpurax/test_recon_cost.py
import time, pytest
from gpurax.recon.adapter import ReconBatch
from gpurax.recon.incremental import preprocess_families

def test_incremental_matches_full(fixture_dir):
    fams = preprocess_families(str(fixture_dir / "species.nwk"),
                               [str(fixture_dir / "gene.nwk")] * 4)
    assert len(fams["families"]) == 4

@pytest.mark.benchmark
def test_preprocess_throughput_recorded(fixture_dir, tmp_path):
    # duplicate the fixture topology N times to simulate a neighbor batch
    paths = [str(fixture_dir / "gene.nwk")] * 200
    t0 = time.perf_counter()
    preprocess_families(str(fixture_dir / "species.nwk"), paths)
    dt = time.perf_counter() - t0
    (tmp_path / "throughput.txt").write_text(f"{200/dt:.1f} topologies/s")
    assert dt < 30.0   # loose guard; real number goes in findings doc
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_recon_cost.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the incremental preprocess path**

```python
# gpurax/recon/incremental.py
import sys, pathlib
_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from gpurec.core.scheduling.batching import preprocess_dataset

def preprocess_families(species_path, topology_paths):
    # Phase-1: use the whole-dataset preprocess as the baseline path.
    # (Incremental build_family_ccp/replan wiring is an optimization gated on the
    #  benchmark below; keep the correct path first.)
    return preprocess_dataset(str(species_path), list(topology_paths))
```

- [ ] **Step 4: Run and record the finding**

Run: `pytest tests/gpurax/test_recon_cost.py -v -m benchmark`
Then write `docs/gpurax/recon_cost_findings.md` with the measured topologies/s, and a decision: if throughput is adequate for expected neighbor-batch sizes, keep `preprocess_dataset`; if not, implement the `build_family_ccp` + `replan_batches` incremental route (reuse unchanged families across steps) and re-measure. **Do not speculate the speedup — record the actual number** (project rule).

- [ ] **Step 5: Commit**

```bash
git add gpurax/recon/incremental.py tests/gpurax/test_recon_cost.py docs/gpurax/recon_cost_findings.md
git commit -m "feat(recon): preprocess-cost probe + incremental-path decision record"
```

### Task C4: Rooting-convention parity check (spec §8 risk)

**Files:**
- Create: `tests/gpurax/test_rooting.py`
- Create: `docs/gpurax/rooting_convention.md`

**Interfaces:**
- Consumes: `recon_loglk` (C1); GeneRax `ReconciliationEvaluation` rooting behavior (`_enforcedRootedGeneTree`, `JointTree.cpp:151`).
- Produces: a documented mapping between gpurec's single-Newick rooting and GeneRax's rooted-GFT handling; if they differ, a `root_gene_tree(newick) -> newick` normalization used by the recon adapter.

- [ ] **Step 1: Write the test comparing rooted vs unrooted inputs**

```python
# tests/gpurax/test_rooting.py
import pytest
from gpurax.recon.parity import recon_loglk

def test_rooting_convention(fixture_dir, generax_ref):
    if not generax_ref.get("available", True):
        pytest.xfail("no GeneRax binary")
    ll = recon_loglk(str(fixture_dir / "species.nwk"), str(fixture_dir / "gene.nwk"),
                     0.2, 0.3, 0.1, device="cpu")
    # GeneRax sums over rootings unless enforced-rooted; document which gpurec does.
    assert ll == pytest.approx(generax_ref["rec_loglk_nats"], rel=1e-4)
```

- [ ] **Step 2: Run and diagnose**

Run: `pytest tests/gpurax/test_rooting.py -v`
If it passes, write `rooting_convention.md` stating "gpurec single-Newick reconciliation matches GeneRax default rooting (verified)". If it fails, determine whether GeneRax marginalizes over rootings while gpurec fixes the input root (inspect `ReconciliationEvaluation` and `crates/gpurec-preprocess` root handling), document the difference, and add the smallest normalization that reconciles them.

- [ ] **Step 3: Commit**

```bash
git add tests/gpurax/test_rooting.py docs/gpurax/rooting_convention.md
git commit -m "test(recon): rooting-convention parity check + documentation"
```

---

## Milestone D — Joint scoring + batch-native SPR search

### Task D1: JointScorer (seq threaded + rec batched)

**Files:**
- Create: `gpurax/search/__init__.py`, `gpurax/search/scorer.py`
- Create: `tests/gpurax/test_scorer.py`

**Interfaces:**
- Consumes: `SeqFamily.seq_loglk` (B1); `ReconBatch.score` (C2).
- Produces: `JointScorer(species_path, model="GTR", rec_weight=1.0, n_threads=None, device="cuda")` with `score_candidates(candidates: list[Candidate]) -> list[float]` where `Candidate = namedtuple("Candidate", "seqfam newick_path rates")`; returns `joint = seq_loglk + rec_weight * rec_loglk` (nats) per candidate. Seq terms computed on a `ThreadPoolExecutor`, rec terms in one `ReconBatch.score` call.

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_scorer.py
import pathlib, pytest
from gpurax.search.scorer import JointScorer, Candidate
from gpurax._seqlik import SeqFamily
FX = pathlib.Path(__file__).parent / "fixtures"

def test_joint_is_seq_plus_recweight_rec():
    newick = (FX / "gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(FX / "aln.fasta"), "GTR")
    sc = JointScorer(str(FX / "species.nwk"), rec_weight=1.0, device="cpu")
    cand = Candidate(seqfam=fam, newick_path=str(FX / "gene.nwk"), rates=(0.2, 0.3, 0.1))
    joint = sc.score_candidates([cand])[0]
    seq = fam.seq_loglk(False)
    from gpurax.recon.parity import recon_loglk
    rec = recon_loglk(str(FX / "species.nwk"), str(FX / "gene.nwk"), 0.2, 0.3, 0.1, device="cpu")
    assert joint == pytest.approx(seq + rec, rel=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_scorer.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `JointScorer`**

```python
# gpurax/search/scorer.py
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from gpurax.recon.adapter import ReconBatch

Candidate = namedtuple("Candidate", "seqfam newick_path rates")

class JointScorer:
    def __init__(self, species_path, model="GTR", rec_weight=1.0, n_threads=None, device="cuda"):
        self._rec = ReconBatch(species_path, device=device)
        self._rw = rec_weight
        self._pool = ThreadPoolExecutor(max_workers=n_threads)

    def score_candidates(self, candidates):
        seqs = list(self._pool.map(lambda c: c.seqfam.seq_loglk(False), candidates))
        recs = self._rec.score([c.newick_path for c in candidates],
                               [c.rates for c in candidates])
        return [s + self._rw * r for s, r in zip(seqs, recs)]
```

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_scorer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/search/ tests/gpurax/test_scorer.py
git commit -m "feat(search): JointScorer combining threaded seq + batched rec terms"
```

### Task D2: Single-family SPR round (greedy best move)

**Files:**
- Create: `gpurax/search/spr.py`
- Create: `tests/gpurax/test_spr_round.py`

**Interfaces:**
- Consumes: `SeqFamily.spr_neighbors/apply_spr/rollback/newick` (B2); `JointScorer.score_candidates` (D1). To materialize each neighbor's topology as a file for the recon adapter: apply the move, write `SeqFamily.newick()` to a temp file, rollback.
- Produces: `spr_round(fam: SeqFamily, rates, scorer, radius, tmpdir, current_joint) -> tuple[bool, float]` — applies the single best joint-improving move (permanently) and returns `(improved, new_joint)`.

- [ ] **Step 1: Write the failing test (monotonic improvement)**

```python
# tests/gpurax/test_spr_round.py
import pathlib
from gpurax._seqlik import SeqFamily
from gpurax.search.scorer import JointScorer
from gpurax.search.spr import spr_round
FX = pathlib.Path(__file__).parent / "fixtures"

def test_round_never_decreases_joint(tmp_path):
    newick = (FX / "gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(FX / "aln.fasta"), "GTR")
    sc = JointScorer(str(FX / "species.nwk"), device="cpu")
    from gpurax.search.scorer import Candidate
    j0 = sc.score_candidates([Candidate(fam, str(FX / "gene.nwk"), (0.2, 0.3, 0.1))])[0]
    improved, j1 = spr_round(fam, (0.2, 0.3, 0.1), sc, radius=1, tmpdir=tmp_path, current_joint=j0)
    assert j1 >= j0 - 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_spr_round.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `spr_round`**

```python
# gpurax/search/spr.py
import pathlib
from gpurax.search.scorer import Candidate

def _materialize(fam, p, r, tmpdir, tag):
    fam.apply_spr(p, r)
    path = str(pathlib.Path(tmpdir) / f"cand_{tag}.nwk")
    pathlib.Path(path).write_text(fam.newick() + "\n")
    fam.rollback()
    return path

def spr_round(fam, rates, scorer, radius, tmpdir, current_joint):
    moves = fam.spr_neighbors(radius)
    if not moves:
        return False, current_joint
    cands, keys = [], []
    for i, (p, r, _path) in enumerate(moves):
        nwk = _materialize(fam, p, r, tmpdir, i)
        cands.append(Candidate(fam, nwk, rates)); keys.append((p, r))
    scores = scorer.score_candidates(cands)
    best = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best] > current_joint:
        p, r = keys[best]
        fam.apply_spr(p, r)          # apply permanently (no rollback)
        return True, scores[best]
    return False, current_joint
```
Note: the seq term for each candidate is computed by `SeqFamily.seq_loglk` on the *current* tree state; since `_materialize` rolls back, D1's scorer must evaluate seq on the candidate topology. Adjust `JointScorer` to accept a per-candidate seq value computed at materialization time (compute `seq_loglk` while the move is applied, store it on the Candidate) — update `Candidate` to `namedtuple("Candidate", "seq_loglk newick_path rates")` and have `_materialize` capture `fam.seq_loglk(False)` between apply and rollback. Update D1's `score_candidates` to read `c.seq_loglk` instead of calling the pool. (Make this refactor here and update `test_scorer.py` accordingly.)

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_spr_round.py tests/gpurax/test_scorer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/search/spr.py gpurax/search/scorer.py tests/gpurax/test_spr_round.py tests/gpurax/test_scorer.py
git commit -m "feat(search): single-family greedy SPR round with candidate materialization"
```

### Task D3: Batch-native multi-family ragged SPR loop

**Files:**
- Modify: `gpurax/search/spr.py`
- Create: `tests/gpurax/test_spr_multi.py`

**Interfaces:**
- Consumes: `spr_round` building blocks (D2), but batched: collect candidates across ALL active families and score in one `JointScorer.score_candidates` call.
- Produces: `spr_search(families: list[FamilyState], scorer, max_radius, tmpdir, *, warmup=False) -> None` mutating each `FamilyState.seqfam` in place. `FamilyState = namedtuple("FamilyState", "name seqfam rates")`. Advances all families one accepted move at a time per radius; retires converged families; increments radius 1..max_radius.

- [ ] **Step 1: Write the equivalence test (batched == per-family)**

```python
# tests/gpurax/test_spr_multi.py
import pathlib, copy
from gpurax._seqlik import SeqFamily
from gpurax.search.scorer import JointScorer
from gpurax.search.spr import spr_search, FamilyState
FX = pathlib.Path(__file__).parent / "fixtures"

def _fresh():
    nwk = (FX / "gene.nwk").read_text().strip()
    return SeqFamily(nwk, str(FX / "aln.fasta"), "GTR")

def test_two_families_converge(tmp_path):
    fams = [FamilyState("f1", _fresh(), (0.2, 0.3, 0.1)),
            FamilyState("f2", _fresh(), (0.2, 0.3, 0.1))]
    sc = JointScorer(str(FX / "species.nwk"), device="cpu")
    spr_search(fams, sc, max_radius=2, tmpdir=tmp_path)
    for fs in fams:            # search terminates with a valid tree per family
        assert fs.seqfam.newick().count("(") >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_spr_multi.py -v`
Expected: FAIL — `spr_search`/`FamilyState` not defined.

- [ ] **Step 3: Implement `spr_search`**

```python
# add to gpurax/search/spr.py
from collections import namedtuple
FamilyState = namedtuple("FamilyState", "name seqfam rates")

def spr_search(families, scorer, max_radius, tmpdir, *, warmup=False):
    for radius in range(1, max_radius + 1):
        active = list(families)
        while active:
            # 1) build candidate batch across all active families
            batch, owner = [], []
            for fi, fs in enumerate(active):
                moves = fs.seqfam.spr_neighbors(radius)
                for i, (p, r, _path) in enumerate(moves):
                    fs.seqfam.apply_spr(p, r)
                    nwk = str(pathlib.Path(tmpdir) / f"r{radius}_f{fi}_{i}.nwk")
                    pathlib.Path(nwk).write_text(fs.seqfam.newick() + "\n")
                    seq = 0.0 if warmup else fs.seqfam.seq_loglk(False)
                    fs.seqfam.rollback()
                    batch.append(Candidate(seq, nwk, fs.rates)); owner.append((fi, p, r))
            if not batch:
                break
            # 2) one batched score call (rec on GPU, seq already captured)
            scores = scorer.score_candidates(batch)
            # 3) per family, pick best; apply if it improves current joint
            best_per = {}
            for k, s in enumerate(scores):
                fi = owner[k][0]
                if fi not in best_per or s > best_per[fi][0]:
                    best_per[fi] = (s, owner[k][1], owner[k][2])
            still = []
            for fi, fs in enumerate(active):
                cur = _current_joint(fs, scorer, warmup)
                s, p, r = best_per[fi]
                if s > cur + 1e-9:
                    fs.seqfam.apply_spr(p, r); still.append(fs)
            active = still

def _current_joint(fs, scorer, warmup):
    seq = 0.0 if warmup else fs.seqfam.seq_loglk(False)
    # score the current (unmoved) topology by writing it once
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkstemp(suffix=".nwk")[1]); p.write_text(fs.seqfam.newick() + "\n")
    return scorer.score_candidates([Candidate(seq, str(p), fs.rates)])[0]
```
During `warmup=True` the seq term is forced to 0.0 → pure reconciliation-only SPR (mirrors GeneRax `--rec-radius`, `enableLibpll=false`). This preserves GeneRax's greedy "apply single best move per family, re-enumerate, repeat" trajectory; batching only changes *when* families synchronize (spec §4).

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_spr_multi.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/search/spr.py tests/gpurax/test_spr_multi.py
git commit -m "feat(search): batch-native ragged multi-family SPR loop with warm-up mode"
```

---

## Milestone E — DTL rate optimization

### Task E1: Global rate optimization adapter

**Files:**
- Create: `gpurax/rates/__init__.py`, `gpurax/rates/optimize.py`
- Create: `tests/gpurax/test_rates.py`

**Interfaces:**
- Consumes: `gpurec.optim.optimize.optimize(...)` and `final_eval` (`gpurec/optim/optimize.py:470,516`); `GeneReconModel` batch_statics.
- Produces: `optimize_global_rates(species_path, tree_paths, init=(0.2,0.3,0.1), device="cuda") -> tuple[float,float,float]` — fitted `(D,L,T)` maximizing summed reconciliation likelihood at fixed trees.

- [ ] **Step 1: Write the failing test (NLL decreases from a bad init)**

```python
# tests/gpurax/test_rates.py
import pathlib
from gpurax.rates.optimize import optimize_global_rates
from gpurax.recon.parity import recon_loglk
FX = pathlib.Path(__file__).parent / "fixtures"

def test_fit_improves_recon(tmp_path):
    sp, gt = str(FX / "species.nwk"), str(FX / "gene.nwk")
    bad = recon_loglk(sp, gt, 0.9, 0.9, 0.9, device="cpu")
    D, L, T = optimize_global_rates(sp, [gt], init=(0.9, 0.9, 0.9), device="cpu")
    fit = recon_loglk(sp, gt, D, L, T, device="cpu")
    assert fit >= bad - 1e-6
    assert all(x > 0 for x in (D, L, T))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_rates.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `optimize_global_rates`**

```python
# gpurax/rates/optimize.py
import math, sys, pathlib, torch
_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from gpurec.api.model import GeneReconModel
from gpurec.optim.optimize import optimize, final_eval

def optimize_global_rates(species_path, tree_paths, init=(0.2, 0.3, 0.1), device="cuda"):
    model = GeneReconModel(species_path, list(tree_paths), mode="global",
                           device=device, dtype=torch.float64)
    theta0 = torch.tensor([math.log2(init[0]), math.log2(init[1]), math.log2(init[2])],
                          dtype=model.theta.dtype, device=model.theta.device)
    theta_fit = optimize(model.batch_statics, theta0)   # returns fitted log2 theta
    with torch.no_grad():
        model.theta.copy_(theta_fit)
    D, L, T = (2.0 ** float(theta_fit[i]) for i in range(3))
    return D, L, T
```
Confirm `optimize`'s exact return/args against `gpurec/optim/optimize.py:470` (it may take `theta0=` kw and return a result object — adapt to extract the fitted log2 vector; use `final_eval` to sanity-check the fitted loss). If `optimize` requires `receiver_weights`, pass `model.receiver_weights`.

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_rates.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/rates/ tests/gpurax/test_rates.py
git commit -m "feat(rates): global DTL-rate optimization adapter over gpurec.optim"
```

---

## Milestone F — I/O and initial trees

### Task F1: GeneRax families-file + species-tree + mapping parsing

**Files:**
- Create: `gpurax/io/__init__.py`, `gpurax/io/families.py`
- Create: `tests/gpurax/test_io.py`

**Interfaces:**
- Produces: `Family = namedtuple("Family", "name starting_tree alignment mapping model")`; `parse_families(path: str) -> list[Family]`; `parse_mapping(path: str) -> dict[str,str]` (gene→species).

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_io.py
import pathlib
from gpurax.io.families import parse_families, parse_mapping
FX = pathlib.Path(__file__).parent / "fixtures"

def test_parse_families():
    fams = parse_families(str(FX / "families.txt"))
    assert len(fams) == 1
    f = fams[0]
    assert f.name == "fam1" and f.model == "GTR"
    assert f.alignment.endswith("aln.fasta")

def test_parse_mapping():
    m = parse_mapping(str(FX / "map.link"))
    assert m == {"a_A": "A", "b_B": "B", "c_C": "C", "d_D": "D"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_io.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement parsers**

```python
# gpurax/io/families.py
from collections import namedtuple
Family = namedtuple("Family", "name starting_tree alignment mapping model")

def parse_families(path):
    fams, cur = [], None
    for line in open(path):
        s = line.strip()
        if not s or s == "[FAMILIES]":
            continue
        if s.startswith("- "):
            if cur:
                fams.append(_finish(cur))
            cur = {"name": s[2:].strip()}
        elif "=" in s and cur is not None:
            k, v = (x.strip() for x in s.split("=", 1))
            cur[{"starting_gene_tree": "starting_tree", "alignment": "alignment",
                 "mapping": "mapping", "subst_model": "model"}.get(k, k)] = v
    if cur:
        fams.append(_finish(cur))
    return fams

def _finish(d):
    return Family(d["name"], d.get("starting_tree"), d.get("alignment"),
                  d.get("mapping"), d.get("model", "GTR"))

def parse_mapping(path):
    m = {}
    for line in open(path):
        s = line.split()
        if len(s) == 2:
            m[s[0]] = s[1]
    return m
```
Confirm mapping format against GeneRax's `IO/GeneSpeciesMapping` — GeneRax also supports `species:gene1;gene2` blocks; add that branch if the parser file shows it and any target dataset uses it.

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_io.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/io/ tests/gpurax/test_io.py
git commit -m "feat(io): GeneRax families-file and mapping parsing"
```

### Task F2: Step 0 — build initial trees when absent

**Files:**
- Create: `gpurax/io/initial_trees.py`
- Create: `tests/gpurax/test_initial_trees.py`

**Interfaces:**
- Consumes: `LibpllEvaluation::createAndSaveRandomTree(aln, model, outfile)` (`LibpllEvaluation.hpp:82`) — exposed via `_seqlik` — as the "build a starting tree from the alignment" primitive (parsimony/random start, then the joint search improves it; this is GeneRax Step 0's role).
- Produces: `ensure_starting_tree(family: Family, workdir: str) -> str` returning a path to a starting Newick, building one via `_seqlik.build_starting_tree(alignment, model, out)` if `family.starting_tree` is None.

- [ ] **Step 1: Expose `build_starting_tree` in `_seqlik`**

Add to seqlik.cpp module block:
```cpp
  m.def("build_starting_tree", [](const std::string &aln, const std::string &model,
                                  const std::string &out) {
    LibpllEvaluation::createAndSaveRandomTree(aln, model, out);
    return out;
  });
```
Rebuild: `pip install -e . --no-build-isolation 2>&1 | tail -3`.

- [ ] **Step 2: Write the failing test**

```python
# tests/gpurax/test_initial_trees.py
import pathlib
from gpurax.io.families import Family
from gpurax.io.initial_trees import ensure_starting_tree
FX = pathlib.Path(__file__).parent / "fixtures"

def test_builds_when_absent(tmp_path):
    fam = Family("f", None, str(FX / "aln.fasta"), str(FX / "map.link"), "GTR")
    tree = ensure_starting_tree(fam, str(tmp_path))
    txt = pathlib.Path(tree).read_text()
    assert txt.count("(") >= 1 and ";" in txt

def test_uses_provided_when_present(tmp_path):
    fam = Family("f", str(FX / "gene.nwk"), str(FX / "aln.fasta"), str(FX / "map.link"), "GTR")
    assert ensure_starting_tree(fam, str(tmp_path)) == str(FX / "gene.nwk")
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/gpurax/test_initial_trees.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement `ensure_starting_tree`**

```python
# gpurax/io/initial_trees.py
import pathlib
from gpurax import _seqlik

def ensure_starting_tree(family, workdir):
    if family.starting_tree:
        return family.starting_tree
    out = str(pathlib.Path(workdir) / f"{family.name}.start.nwk")
    return _seqlik.build_starting_tree(family.alignment, family.model, out)
```

- [ ] **Step 5: Run**

Run: `pytest tests/gpurax/test_initial_trees.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gpurax/_seqlik/seqlik.cpp gpurax/io/initial_trees.py tests/gpurax/test_initial_trees.py
git commit -m "feat(io): Step-0 initial-tree building via LibpllEvaluation random/parsimony tree"
```

---

## Milestone G — Reconcile + export

### Task G1: Final reconciliation via gpurec backtracking

**Files:**
- Create: `gpurax/reconcile/__init__.py`, `gpurax/reconcile/sample.py`
- Create: `tests/gpurax/test_reconcile.py`

**Interfaces:**
- Consumes: `sample_reconciliations(model, *, family_index, seed)` (`gpurec/core/backtracking/input.py:37`) returning a list of `(event: str, species_index: int, [left_idx, right_idx])`; event strings `speciation|duplication|transfer|leaf|loss`.
- Produces: `reconcile_family(species_path, tree_path, rates, *, seed=0, device="cuda") -> list[tuple]` (the sampled reconciliation node list).

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_reconcile.py
import pathlib
from gpurax.reconcile.sample import reconcile_family
FX = pathlib.Path(__file__).parent / "fixtures"

def test_reconcile_returns_events():
    nodes = reconcile_family(str(FX / "species.nwk"), str(FX / "gene.nwk"),
                             (0.2, 0.3, 0.1), seed=0, device="cpu")
    assert len(nodes) >= 1
    kinds = {n[0] for n in nodes}
    assert kinds <= {"speciation", "duplication", "transfer", "leaf", "loss"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_reconcile.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `reconcile_family`**

```python
# gpurax/reconcile/sample.py
import math, sys, pathlib, torch
_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from gpurec.api.model import GeneReconModel
from gpurec import sample_reconciliations

def reconcile_family(species_path, tree_path, rates, *, seed=0, device="cuda"):
    model = GeneReconModel(species_path, [tree_path], mode="global",
                           device=device, dtype=torch.float64)
    triple = torch.tensor([math.log2(rates[0]), math.log2(rates[1]), math.log2(rates[2])],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        model.theta.copy_(triple)
    return sample_reconciliations(model, family_index=0, seed=seed)
```

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_reconcile.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/reconcile/ tests/gpurax/test_reconcile.py
git commit -m "feat(reconcile): final reconciliation via gpurec backtracking"
```

### Task G2: Export — Newick + scores.tsv

**Files:**
- Create: `gpurax/reconcile/export.py`
- Create: `tests/gpurax/test_export_basic.py`

**Interfaces:**
- Produces: `write_newick(out_dir, name, newick_str)`; `write_scores(out_dir, rows: list[dict])` where each row has `name, joint, rec, seq` → `scores.tsv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_export_basic.py
from gpurax.reconcile.export import write_newick, write_scores

def test_writes_newick_and_scores(tmp_path):
    write_newick(str(tmp_path), "fam1", "((a,b),(c,d));")
    write_scores(str(tmp_path), [{"name": "fam1", "joint": -10.0, "rec": -3.0, "seq": -7.0}])
    assert (tmp_path / "fam1.reconciled.nwk").read_text().strip() == "((a,b),(c,d));"
    tsv = (tmp_path / "scores.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ["name", "joint", "rec", "seq"]
    assert tsv[1].split("\t")[0] == "fam1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_export_basic.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement exporters**

```python
# gpurax/reconcile/export.py
import pathlib

def write_newick(out_dir, name, newick_str):
    p = pathlib.Path(out_dir) / f"{name}.reconciled.nwk"
    p.write_text(newick_str.strip() + "\n")
    return str(p)

def write_scores(out_dir, rows):
    p = pathlib.Path(out_dir) / "scores.tsv"
    cols = ["name", "joint", "rec", "seq"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in cols))
    p.write_text("\n".join(lines) + "\n")
    return str(p)
```

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_export_basic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/reconcile/export.py tests/gpurax/test_export_basic.py
git commit -m "feat(export): reconciled Newick + scores.tsv"
```

### Task G3: Export — RecPhyloXML

**Files:**
- Modify: `gpurax/reconcile/export.py`
- Create: `tests/gpurax/test_export_recphyloxml.py`

**Interfaces:**
- Consumes: the reconciliation node list from G1 (`(event, species_index, [l, r])`), plus the species tree (for `<clade>` species names).
- Produces: `write_recphyloxml(out_dir, name, gene_newick, recon_nodes, species_newick) -> str` writing valid RecPhyloXML (`<recGeneTree>` with per-node `<eventsRec>` tags: `<leaf>/<speciation>/<duplication>/<transferBack>/<loss>`).

- [ ] **Step 1: Write the failing test (well-formed XML with event tags)**

```python
# tests/gpurax/test_export_recphyloxml.py
import xml.dom.minidom as md
from gpurax.reconcile.export import write_recphyloxml

def test_recphyloxml_is_valid(tmp_path):
    nodes = [("speciation", 4, [1, 2]), ("leaf", 0, [-1, -1]), ("leaf", 1, [-1, -1])]
    out = write_recphyloxml(str(tmp_path), "fam1", "((a,b),c);", nodes, "((A,B),C);")
    doc = md.parse(out)                       # raises if malformed
    tags = {n.tagName for n in doc.getElementsByTagName("*")}
    assert "recPhylo" in tags and "eventsRec" in tags
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_export_recphyloxml.py -v`
Expected: FAIL — `write_recphyloxml` not defined.

- [ ] **Step 3: Implement RecPhyloXML export**

Build the XML with `xml.etree.ElementTree`: a `<recPhylo>` root containing `<spTree>` (from `species_newick`, converted to phyloXML `<clade>` nesting) and `<recGeneTree>` (from `gene_newick` + `recon_nodes`, each `<clade>` carrying an `<eventsRec>` child whose event element is chosen from the node's event string: `leaf`→`<leaf speciesLocation=...>`, `speciation`→`<speciation ...>`, `duplication`→`<duplication ...>`, `transfer`→`<transferBack destinationSpecies=...>`, `loss`→`<loss ...>`). Map `species_index` to species names via a preorder labeling of `species_newick`. Reference the RecPhyloXML spec element names in `ext/GeneRaxCore/src/IO/` (GeneRax's own RecPhyloXML writer) to match tag/attribute names exactly. Write with `ElementTree.indent` + `write(out, xml_declaration=True, encoding="unicode")`.

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_export_recphyloxml.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/reconcile/export.py tests/gpurax/test_export_recphyloxml.py
git commit -m "feat(export): RecPhyloXML reconciliation output"
```

### Task G4: Export — Notung

**Files:**
- Modify: `gpurax/reconcile/export.py`
- Create: `tests/gpurax/test_export_notung.py`

**Interfaces:**
- Produces: `write_notung(out_dir, name, gene_newick, recon_nodes) -> str` writing a Notung-style annotated tree (NHX-tagged Newick with `[&&NHX:...]` event/host annotations, matching GeneRax's Notung writer conventions).

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_export_notung.py
from gpurax.reconcile.export import write_notung

def test_notung_has_nhx_and_events(tmp_path):
    nodes = [("duplication", 4, [1, 2]), ("leaf", 0, [-1, -1]), ("leaf", 1, [-1, -1])]
    out = write_notung(str(tmp_path), "fam1", "((a,b),c);", nodes)
    txt = open(out).read()
    assert "[&&NHX" in txt and txt.strip().endswith(";")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_export_notung.py -v`
Expected: FAIL — `write_notung` not defined.

- [ ] **Step 3: Implement Notung export**

Emit the gene Newick with per-internal-node NHX annotations (`[&&NHX:S=<species>:D=Y]` for duplications, host/event tags per Notung's format). Reuse the preorder species labeling from G3. Match tag conventions to GeneRax's Notung writer in `ext/GeneRaxCore/src/IO/` (open it to copy exact NHX keys). Traverse `recon_nodes` (node 0 = root) to attach the annotation to the matching gene-tree clade.

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_export_notung.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/reconcile/export.py tests/gpurax/test_export_notung.py
git commit -m "feat(export): Notung reconciliation output"
```

---

## Milestone H — Orchestration driver + CLI

### Task H1: Driver — Step0 → warm-up → (rates ↔ SPR) → reconcile → export

**Files:**
- Create: `gpurax/driver.py`
- Create: `tests/gpurax/test_driver.py`

**Interfaces:**
- Consumes: all of F1/F2 (I/O + Step 0), D1–D3 (scorer + SPR), E1 (rates), G1–G4 (reconcile + export).
- Produces: `run(families_path, species_path, out_dir, *, rec_model="UndatedDTL", max_spr_radius=5, rec_radius=1, rec_weight=1.0, device="cuda") -> None` implementing the pipeline from spec §3.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/gpurax/test_driver.py
import pathlib
from gpurax.driver import run
FX = pathlib.Path(__file__).parent / "fixtures"

def test_end_to_end_small(tmp_path):
    run(str(FX / "families.txt"), str(FX / "species.nwk"), str(tmp_path),
        max_spr_radius=2, rec_radius=1, device="cpu")
    assert (tmp_path / "scores.tsv").exists()
    assert (tmp_path / "fam1.reconciled.nwk").exists()
    assert (tmp_path / "fam1.recphyloxml").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_driver.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `run`**

Wire the pipeline (spec §3): `parse_families` → `ensure_starting_tree` per family → build a `SeqFamily` per family → **warm-up** `spr_search(..., warmup=True, max_radius=rec_radius)` → **main loop** `for r in 1..max_spr_radius:` `optimize_global_rates` (E1) then `spr_search(..., warmup=False, max_radius=r)` → per family `reconcile_family` (G1) → `write_newick`/`write_recphyloxml`/`write_notung`/`write_scores`. Compute each family's final `joint/rec/seq` for `scores.tsv` via the scorer + `recon_loglk`. Use a `tempfile.TemporaryDirectory()` for candidate materialization. Keep the function under ~60 lines; push any helper longer than a few lines into `search/` or `reconcile/`.

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_driver.py -v`
Expected: PASS (all output files present).

- [ ] **Step 5: Commit**

```bash
git add gpurax/driver.py tests/gpurax/test_driver.py
git commit -m "feat(driver): end-to-end Step0->warmup->rates/SPR->reconcile->export pipeline"
```

### Task H2: CLI mirroring GeneRax

**Files:**
- Create: `gpurax/cli.py`
- Modify: `gpurax/__init__.py` (export `main`)
- Create: `tests/gpurax/test_cli.py`
- Modify: `pyproject.toml` (console_scripts entry `gpurax = gpurax.cli:main`)

**Interfaces:**
- Consumes: `gpurax.driver.run`.
- Produces: `main(argv: list[str] | None = None) -> int`; CLI flags `-f/--families -s/--species-tree -r/--rec-model --max-spr-radius --rec-radius --rec-weight -p/--prefix --device`.

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_cli.py
import pathlib
from gpurax.cli import main
FX = pathlib.Path(__file__).parent / "fixtures"

def test_cli_runs(tmp_path):
    rc = main(["-f", str(FX / "families.txt"), "-s", str(FX / "species.nwk"),
               "-r", "UndatedDTL", "--max-spr-radius", "2", "-p", str(tmp_path),
               "--device", "cpu"])
    assert rc == 0
    assert (tmp_path / "scores.tsv").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `main`**

```python
# gpurax/cli.py
import argparse
from gpurax.driver import run

def main(argv=None):
    p = argparse.ArgumentParser(prog="gpurax")
    p.add_argument("-f", "--families", required=True)
    p.add_argument("-s", "--species-tree", required=True)
    p.add_argument("-r", "--rec-model", default="UndatedDTL")
    p.add_argument("--max-spr-radius", type=int, default=5)
    p.add_argument("--rec-radius", type=int, default=1)
    p.add_argument("--rec-weight", type=float, default=1.0)
    p.add_argument("-p", "--prefix", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    run(a.families, a.species_tree, a.prefix, rec_model=a.rec_model,
        max_spr_radius=a.max_spr_radius, rec_radius=a.rec_radius,
        rec_weight=a.rec_weight, device=a.device)
    return 0
```

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/cli.py gpurax/__init__.py pyproject.toml tests/gpurax/test_cli.py
git commit -m "feat(cli): GeneRax-mirroring command-line interface"
```

---

## Milestone I — Validation

### Task I1: Integration parity vs GeneRax (joint-loglk + RF)

**Files:**
- Create: `tests/gpurax/test_parity_integration.py`

**Interfaces:**
- Consumes: `gpurax.driver.run`; `generax_ref.json` (A1); an RF function (dendropy `TreeList`/`symmetric_difference` or ete3 `compare`).

- [ ] **Step 1: Write the integration test**

```python
# tests/gpurax/test_parity_integration.py
import json, pathlib, pytest
from gpurax.driver import run
FX = pathlib.Path(__file__).parent / "fixtures"

def _rf(nwk_a, nwk_b):
    import dendropy
    tns = dendropy.TaxonNamespace()
    a = dendropy.Tree.get(data=nwk_a, schema="newick", taxon_namespace=tns)
    b = dendropy.Tree.get(data=nwk_b, schema="newick", taxon_namespace=tns)
    return dendropy.calculate.treecompare.symmetric_difference(a, b)

def test_joint_and_rf_parity(tmp_path, generax_ref):
    if not generax_ref.get("available", True):
        pytest.xfail("no GeneRax binary")
    run(str(FX / "families.txt"), str(FX / "species.nwk"), str(tmp_path),
        max_spr_radius=3, device="cpu")
    rows = (tmp_path / "scores.tsv").read_text().splitlines()
    joint = float(dict(zip(rows[0].split("\t"), rows[1].split("\t")))["joint"])
    assert joint == pytest.approx(generax_ref["joint_loglk_nats"], rel=1e-2)
    ours = (tmp_path / "fam1.reconciled.nwk").read_text()
    assert _rf(ours, generax_ref["reconciled_newick"]) <= 2   # low RF on 4 taxa
```

- [ ] **Step 2: Run**

Run: `pytest tests/gpurax/test_parity_integration.py -v`
Expected: PASS or xfail. If joint differs beyond tolerance, decompose: compare per-family `rec` (C1 parity) and `seq` (B1 parity) columns separately to localize the discrepancy.

- [ ] **Step 3: Commit**

```bash
git add tests/gpurax/test_parity_integration.py
git commit -m "test(gpurax): integration parity vs GeneRax (joint-loglk + RF)"
```

### Task I2: Regression slice (cyanobacteria / primates)

**Files:**
- Create: `tests/gpurax/test_regression_slice.py`
- Create: `docs/gpurax/validation.md`

**Interfaces:**
- Consumes: a small slice (e.g. 5–10 families) of the paper's cyanobacteria or primates dataset (locate under `experiments/` or the paper's `generax_data.tar.gz`).

- [ ] **Step 1: Locate a real multi-family dataset slice**

Run:
```bash
find /home/enzo/Documents/git/gpurec/consolidate-release/experiments -iname '*.fasta' -path '*cyano*' 2>/dev/null | head
find /home/enzo/Documents/git/gpurec/consolidate-release -iname 'families*.txt' 2>/dev/null | head
```
Assemble a families file for ~5 families into `tests/gpurax/fixtures/slice/`. If none is present locally, document in `validation.md` that the slice must be fetched from the paper's `generax_data.tar.gz` and mark the test `skip(reason=...)`.

- [ ] **Step 2: Write the regression test**

```python
# tests/gpurax/test_regression_slice.py
import pathlib, pytest
from gpurax.driver import run
SLICE = pathlib.Path(__file__).parent / "fixtures" / "slice"

@pytest.mark.slow
def test_slice_runs_and_scores(tmp_path):
    if not (SLICE / "families.txt").exists():
        pytest.skip("no dataset slice available (see docs/gpurax/validation.md)")
    run(str(SLICE / "families.txt"), str(SLICE / "species.nwk"), str(tmp_path),
        max_spr_radius=3, device="cuda")
    rows = (tmp_path / "scores.tsv").read_text().splitlines()
    assert len(rows) >= 6                       # header + >=5 families
    assert all(len(r.split("\t")) == 4 for r in rows)
```

- [ ] **Step 3: Run**

Run: `pytest tests/gpurax/test_regression_slice.py -v -m slow`
Expected: PASS or skip. Record the run's joint log-likelihoods and wall-time in `validation.md`; compare against GeneRax on the same slice if a binary is available. **Do not run large/long benchmarks without asking the user first** (project rule) — keep the slice to ≤10 families.

- [ ] **Step 4: Commit**

```bash
git add tests/gpurax/test_regression_slice.py docs/gpurax/validation.md tests/gpurax/fixtures/slice/
git commit -m "test(gpurax): regression slice + validation record"
```

### Task I3: Seq-vs-recon timing instrumentation (Phase-2 trigger)

**Files:**
- Modify: `gpurax/search/scorer.py` (add timing accumulators)
- Create: `tests/gpurax/test_timing.py`

**Interfaces:**
- Produces: `JointScorer.timings -> dict{"seq_s": float, "rec_s": float}` accumulated across `score_candidates` calls; the driver logs the ratio at the end.

- [ ] **Step 1: Write the failing test**

```python
# tests/gpurax/test_timing.py
import pathlib
from gpurax.search.scorer import JointScorer, Candidate
FX = pathlib.Path(__file__).parent / "fixtures"

def test_timings_accumulate():
    sc = JointScorer(str(FX / "species.nwk"), device="cpu")
    sc.score_candidates([Candidate(-5.0, str(FX / "gene.nwk"), (0.2, 0.3, 0.1))])
    assert sc.timings["seq_s"] >= 0.0 and sc.timings["rec_s"] > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gpurax/test_timing.py -v`
Expected: FAIL — `timings` attribute missing.

- [ ] **Step 3: Add timing to `JointScorer`**

In `__init__`, set `self.timings = {"seq_s": 0.0, "rec_s": 0.0}`. In `score_candidates`, wrap the seq read and the `self._rec.score(...)` call in `time.perf_counter()` deltas and accumulate. (Seq is now pre-captured on the Candidate, so `seq_s` measures only the read/sum; the meaningful measurement is `rec_s` plus the seq computation cost already paid at materialization — note this in a comment so the Phase-2 decision uses the *materialization* seq cost, which the driver should also time.)

- [ ] **Step 4: Run**

Run: `pytest tests/gpurax/test_timing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurax/search/scorer.py tests/gpurax/test_timing.py
git commit -m "feat(search): seq-vs-recon timing instrumentation for Phase-2 trigger"
```

---

## Full-suite verification

- [ ] Run the whole suite on CPU: `pytest tests/gpurax/ -v` — all PASS/xfail/skip, no ERROR/FAIL.
- [ ] If a GPU is available, re-run the non-parity tests with `device="cuda"` by exporting `GPURAX_TEST_DEVICE=cuda` (wire this env into `conftest.py` as a `device` fixture) to confirm GPU parity.
- [ ] Confirm `docs/gpurax/recon_cost_findings.md`, `rooting_convention.md`, and `validation.md` contain real recorded numbers, not placeholders.

## Notes on later phases (out of scope here)

- **Phase 2 (gated on I3 measurement):** GPU Felsenstein pruning engine if the sequence term dominates.
- Per-species / per-family DTL rates (gpurec supports; add a `--per-species-rates`/`--per-family-rates` path using `fit_genewise`).
- Species-tree search, MPI multi-node scheduling.
