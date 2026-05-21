# Notebook Ownership

The notebooks in this directory are checkout-local HOGENOM analysis artifacts,
not portable examples or supported release workflows.  They assume a source
checkout with the local untracked HOGENOM data layout, CUDA, notebook
dependencies, and the same optional packages used by the HOGENOM experiment
scripts.

Use the installed `gpurec` CLI and the main README examples for supported
workflows.  Before promoting notebook logic into production, move the behavior
into `gpurec.workflow`, add tests, and document the CLI surface first.

| Notebook | Status | Assumptions | Next action |
| --- | --- | --- | --- |
| `reconcile_hogenom_ccp_gpurec.ipynb` | Historical HOGENOM reconciliation analysis. | Local HOGENOM tree/family files, CUDA, and source-tree imports. | Keep as an archived analysis note unless the workflow is migrated into scripts or the CLI. |
| `optimize_hogenom_ccp_adam_oscillation.ipynb` | Historical optimizer-oscillation investigation. | Local HOGENOM data, CUDA, source-tree imports, and ad hoc plotting/state inspection. | Keep as an archived investigation unless converted into a reproducible benchmark or regression test. |

