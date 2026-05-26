# Config Ownership

Configs in this directory are source-checkout files.  They are not installed
package templates unless explicitly documented as such.

## Current Files

| Config | Owner | Format | Portability |
| --- | --- | --- | --- |
| `hogenom_ccp_wandb.yaml` | Legacy HOGENOM Hydra/W&B launcher. | Hydra YAML consumed by `python scripts/optimize_hogenom_ccp_hydra.py`, not by `gpurec optimize --config`. | Checkout-local only: assumes untracked `tests/data/HOGENOM/...` inputs, CUDA, online W&B defaults, and local output/cache directories. |

The installed workflow uses flat JSON `RunConfig` files through `gpurec
optimize --config`, `gpurec sample`, and `gpurec run`.  `examples/README.md`
maps the source-checkout/source-archive JSON parser fixtures for that flat JSON
surface, including `examples/minimal-run-config.json` for genewise
`hessian-sgd` auto defaults and
`examples/specieswise-adagrad-restarts-config.json` for specieswise
`adagrad-restarts` auto defaults. These fixtures are not CPU fallbacks and not
end-to-end optimizer smokes while the retained backward path requires `S > 256`.

New tracked configs should state:

- which command consumes the file;
- whether the file is portable across checkouts;
- required local data, CUDA, W&B, or other external services;
- whether it is a parser fixture, a maintained production template, or a
  historical experiment input.
