# Glossary

This page defines stable user-facing terminology used across the CLI, run
configs, and output artifacts.

- `D`: duplication event rate parameter.
- `T`: transfer event rate parameter.
- `L`: loss event rate parameter.
- `DTL`: shorthand for the three event rates `D`, `T`, and `L`.
- `CCP`: conditional clade probability representation derived from one or more
  gene-tree records for a family.
- `global` mode: one shared `D/T/L` triplet across all families and species.
- `specieswise` mode: one `D/T/L` triplet per species, shared across families.
- `genewise` mode: one `D/T/L` triplet per family, shared across species.
- `RecPhyloXML`: XML format used for reconciliation samples exported by
  `gpurec sample` and `gpurec run`.
- `NLL`: negative log-likelihood objective minimized during optimization.
- `route`: the resolved objective/evaluator/optimizer path reported in
  validation and summary metadata.
- `solver budget`: iteration controls for fixed-point and adjoint solves
  (`fixed_iters_Pi`, `neumann_terms`, and related route settings).
- `checkpoint`: serialized optimization state stored under `checkpoints/`
  (`latest.pt`, `best.pt`, and optional periodic snapshots).
