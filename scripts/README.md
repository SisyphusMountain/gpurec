# HOGENOM Experiment Scripts

The installed `gpurec` CLI is the supported workflow for general optimization,
checkpointing, and sampling:

```bash
gpurec optimize --species-tree S.tree --families-file families.txt --out-dir out
```

The scripts in this directory are checkout-local experiment launchers,
diagnostics, and compatibility helpers.  The large launchers
`hogenom_ccp_wandb_opt.py` and `fast_optimize_hogenom_ccp.py` are legacy
HOGENOM reproducers: they retain experiment-specific optimizer schedules,
plotting, W&B behavior, local path defaults, and reporting conventions.
Some older launchers, including the global-uniform and specieswise-uniform
scripts, are fixed-dataset reproducers with HOGENOM paths declared as module
constants rather than general path flags.

New production workflow behavior should go into `gpurec.workflow` and the
installed CLI first.  Mirror it into these legacy scripts only when a retained
HOGENOM experiment needs that behavior.
