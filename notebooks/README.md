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

## Ignored Local Notebooks

`.gitignore` ignores notebooks by default; only the two notebooks above are
tracked.  The ignored notebooks seen in this checkout are local workspace
artifacts, not release inputs.  Keep their decisions in docs before deleting or
migrating them.

| Ignored notebook | Purpose | Current reproducibility | Decision |
| --- | --- | --- | --- |
| `evaluate_gpurec_at_alerax_params.ipynb` | Evaluates gpurec likelihoods at AleRax global-rate parameters and writes `nll_at_alerax_params_*.csv`. | Requires ignored `tests/data/hogenom_bench`, CUDA, AleRax output files, and notebook state. | Archive/delete or migrate into a tested rate-evaluation script if the comparison remains useful. |
| `hogenom_adam_bfgs_schedule.ipynb` | Experiments with scheduled Adam/BFGS optimization on HOGENOM benchmark inputs. | Imports the historical `gpurec.optimization.optimize_scheduled` helper and uses ignored `tests/data/hogenom_bench`. | Archive/delete or rewrite against supported workflow optimizers before keeping. |
| `optimize_hogenom_ccp_specieswise_origination.ipynb` | One-off specieswise optimization with configurable origination distribution. | Requires local HOGENOM data, CUDA, and contains captured error output from the historical run. | Migrate unique origination behavior into the CLI/workflow or archive/delete. |
| `pi_iteration_bound_diagnostic.ipynb` | Fixed-point Pi iteration bound diagnostic with plots and CSV output. | Imports local `pi_iteration_bound_diagnostic_impl` source that is not tracked; requires ignored HOGENOM benchmark data. | Restore helper source and tests before keeping, otherwise archive/delete. |
