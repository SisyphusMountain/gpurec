# gpurec Source Examples

These files are source-checkout and source-archive fixtures for the production
CLI. They are intentionally tiny so `gpurec validate-config` can exercise flat
JSON config parsing, relative path resolution, AleRax family references, mapping
files, and mode-specific optimizer defaults without constructing the CUDA
likelihood model.

The tiny species tree has too few species for the retained Pi backward path, so
these examples are not end-to-end optimizer smokes. Use them as config
templates, then replace the tree and family paths with a real dataset whose
species tree satisfies the current CUDA backward requirements.

```bash
gpurec validate-config --config examples/minimal-run-config.json
gpurec validate-config --config examples/specieswise-adagrad-restarts-config.json
```

- `minimal-run-config.json` uses `mode=genewise`; `optimizer=auto` resolves to
  `hessian-sgd`.
- `specieswise-adagrad-restarts-config.json` uses `mode=specieswise`;
  `optimizer=auto` resolves to `adagrad-restarts` with the default
  `8:1.0:60,16:0.5:35,32:0.5:30` schedule and fixed128 final validation.

Installed wheels intentionally do not install this directory as package data.
Installed users can generate equivalent flat JSON starting points with:

```bash
gpurec config-template --mode genewise --output run.json
gpurec config-template --mode specieswise --output specieswise-run.json
```

Copy one of these JSON files next to your own `S.tree` and `[FAMILIES]` file, or
pass the same fields directly as CLI flags.
