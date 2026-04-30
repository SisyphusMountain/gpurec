# ALE-Rax stochastic backtracking: how it works (notes)

This is a guided walkthrough of the reconciliation sampling pipeline in `extra/AleRax_modified`, focusing on how stochastic backtracking uses precomputed partial likelihoods to sample reconciliation scenarios.

## Big picture
- Inputs: species tree (possibly dated), ALE conditional clade probabilities (CCP) file per family, gene↔species mapping, D/T/L rates, optional gamma categories, optional transfer highways.
- Precompute per-clade, per-species, per-category “partial likelihoods” (`uq`) and extinction terms (`uE`, `uEBar`) from CCPs and event probabilities.
- Sample a reconciliation by backtracking through the dynamic-programming lattice: at each step, probabilistically pick an event (S, D, T, SL, DL, TL, or leaf) proportional to its contribution to the local CLV/partial.
- Build a reconciled gene tree and event list (`Scenario`), then export counts and files.

Key classes/files:
- Multi-model scaffolding and backtracking driver: `extra/AleRax_modified/src/ale/MultiModel.hpp`
- DTL (with transfers) multi-family model: `extra/AleRax_modified/src/ale/UndatedDTLMultiModel.hpp`
- DL-only multi-family model: `extra/AleRax_modified/src/ale/UndatedDLMultiModel.hpp`
- Scenario storage, export, helpers: `extra/AleRax_modified/ext/GeneRaxCore/src/util/Scenario.hpp`, `.../Scenario.cpp`
- Orchestration over families: `extra/AleRax_modified/src/ale/AleEvaluator.cpp`, `extra/AleRax_modified/src/ale/AleOptimizer.cpp`

## Data flow and setup
1) CCPs and mapping
- CCPs are loaded in `MultiModel` ctor (`ConditionalClades::unserialize`) and mapped to species via leaf name mapping.
- Species tree may be “pruned” to the set of covered species leaves; mapping `_speciesToPrunedNode` is computed to handle that.

2) Rates → event probabilities
- Raw per-species D/T/L rates (and gamma scalers) are converted into normalized event probabilities per branch and gamma category:
  - `_PD[e,c], _PL[e,c], _PT[e,c], _PS[e,c]` in `UndatedDTLMultiModel::recomputeSpeciesProbabilities()`.
  - Origination prior `_OP[e]` depends on `OriginationStrategy` (UNIFORM, ROOT, LCA, OPTIMIZE).

3) Extinction/fixed point terms
- Extinction probabilities `_uE[e,c]` and “other-side extinction” `_uEBar[e,c]` are computed by fixed-point iteration, including TL and highway contributions.
- Transfer constraints alter the sums used during updates: NONE, PARENTS (forbid transfers to ancestors), RELDATED (soft dating order).

4) Per-clade partials (CLVs)
- For each clade id `cid`, compute `_dtlclvs[cid]._uq[e,c]` with dynamic programming over events and CCP splits in `UndatedDTLMultiModel::updateCLV()`.
- Also precompute per-clade transfer mass helpers: `_survivingTransferSum[c]`, `_correctionSum[e,c]`, `_correctionNorm[e,c]` for constraint-aware transfer weights (`getTransferSum`).

DL-only model mirrors this without transfers (`UndatedDLMultiModel`).

## Event contributions (DTL model)
At a given clade `cid`, species node `e`, category `c`, contributions align with the usual D/T/L/S/SL/DL/TL terms against CCP splits `(left,right,freq)`:
- Speciation S (internal species nodes):
  - `uq[left][f,c] * uq[right][g,c] * (PS[e,c] * freq)` and swapped left/right.
- Duplication D: `uq[left][e,c] * uq[right][e,c] * (PD[e,c] * freq)`.
- Transfer T: `getTransferSum(left,e,c) * (PT[e,c] * freq) * uq[right][e,c]` and symmetric.
- Highways: extra T mass to designated destination species (weighted by `highway.proba`).
- Speciation+Loss SL (internal species nodes): `uq[cid][child,c] * (uE[sibling,c] * PS[e,c])` for each child/sibling pairing.
- Duplication+Loss DL: `uq[cid][e,c] * (2*uE[e,c]) * PD[e,c]`.
- Transfer+Loss TL: two cases — TL to dest and lose in source, or transfer out and lose in dest; both add mass, with highways as well.

See `UndatedDTLMultiModel.hpp` around computeProbability and helpers.

## Stochastic backtracking
Implemented in `MultiModelTemplate<REAL>::backtrace` and the model’s `computeProbability`.

1) Root choice
- Choose starting species node and gamma category with weight proportional to `uq[rootCID][e,c] * OP[e]` and conditioned by survival factor; see `sampleSpeciesNode()`.

2) Two-pass local sampling at (cid, species, category)
- Pass 1: call `computeProbability(cid,e,c, proba, recCell=nullptr)` to get total mass `proba` for all admissible events.
- Draw threshold `maxProba = proba * Random::getProba()`.
- Pass 2: call `computeProbability` again with `recCell` to cumulatively sum event masses in the same order. The first event where cumulative mass exceeds `maxProba` is selected; event details are filled into `recCell.event` (including which gene split orientation, destination species for T/TL, branch lengths for children from CCP split if any).

3) Scenario update and recursion
- For S, D, T: `Scenario::generateGeneChildren()` creates two gene children; `Scenario::addEvent()` records the event. Recurse on each child with the appropriate `(cid, species)` pair: S splits to species children; D duplicates within same species; T splits into source species and sampled destination species.
- For SL: keep the gene clade and move to the surviving child species.
- For DL: not recorded as a separate event; continue from the same `(cid,e)` (effectively resampling under a “duplication then loss” step).
- For TL: either resample again at `(cid,e)` if the transfer dies in dest; or move the lineage to the sampled destination species.
- For leaf: assign gene leaf label from CCP when `(cid, species)` are terminal and mapped.

See backtrace flow in `MultiModel.hpp` near `backtrace(...)` and `computeProbability(...)` in model headers.

4) Transfer destination sampling
- `sampleTransferEvent(cid, originSpecies, c, event)` draws a destination species proportional to `uq[cid][dest,c]`, honoring constraints (skip parents or non-chronologically valid recipients), and sets `event.pllDestSpeciesNode`/`destSpeciesNode`.

5) Termination
- Recursion ends when all gene leaves are assigned; the `Scenario` contains the reconciled gene tree nodes and all recorded events, ready for export.

## Scenario, outputs, and summaries
- `Scenario` stores events per gene node; it also tracks last event type to classify singletons, origin, etc. It can output:
  - Reconciliations in NHX, ALE, RecPhyloXML, and Newick-with-events formats.
  - Per-species event counts, transfer matrices, origin statistics, ortho-groups.
- Sampling across families: `AleEvaluator::sampleFamilyScenarios()` computes CLVs for a family, samples N scenarios, and returns them; `AleOptimizer::reconcile()` writes per-sample reconciliations and merged summaries.

## Highways
- Optional transfer “highways” bias transfers along specified source→dest pairs. They are attached in `setHighways()`, normalized alongside event probabilities, and injected in both `computeProbability` (adds explicit transfer mass to the highway destination) and extinction iterations.

## Numerical considerations
- Types: both `double` and high-precision `ScaledValue` are supported; sampling falls back to high precision if needed.
- Conditioning on survival: species-root sampling divides by `getLikelihoodFactor(category)` to condition on lineage survival.
- MPI consistency: stochastic draws are made consistent across ranks by calling `ParallelContext::makeRandConsistent()` at safe points.

## Where to look in code (entry points)
- Backtracking driver and event recursion: `extra/AleRax_modified/src/ale/MultiModel.hpp` (backtrace, sampleReconciliations, _computeScenario)
- DTL model CLV build, event mass, dest sampling: `extra/AleRax_modified/src/ale/UndatedDTLMultiModel.hpp`
- DL-only model: `extra/AleRax_modified/src/ale/UndatedDLMultiModel.hpp`
- Scenario data structure and export: `extra/AleRax_modified/ext/GeneRaxCore/src/util/Scenario.hpp`, `.../Scenario.cpp`
- Orchestrating sampling and writing outputs: `extra/AleRax_modified/src/ale/AleEvaluator.cpp`, `extra/AleRax_modified/src/ale/AleOptimizer.cpp`

## Minimal pseudocode sketch
```
compute_all_CLVs();
(e,c) = sample_species_root_and_category();
init virtual gene root;
backtrace(cid_root, e, gene_root, c):
  total = sum_over_events_masses(cid,e,c);
  thresh = U(0,1) * total;
  cum = 0;
  for event in canonical_order:
    cum += mass(event);
    if cum > thresh:
      record_event_if_needed(event);
      spawn_gene_children_if_S/D/T(event);
      (cid',e') pairs = next_states(event);
      recurse on each (cid',e') with appropriate gene child;
      break;
```

That’s the core of “stochastic backtracking” on top of ALE partials: use the dynamic program as a proposal distribution and walk it randomly to realize one reconciliation scenario.

