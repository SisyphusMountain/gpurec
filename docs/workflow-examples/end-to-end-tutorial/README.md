# End-to-End Tutorial (Public, Minimal Dataset)

This workflow uses a redistributable synthetic dataset that is intentionally
small and deterministic. The species tree has >256 taxa so it can pass the
`S > 256` CUDA backward gate once CUDA is available.

This is the first successful run tutorial and uses only public commands
(`gpurec ...`) plus the tracked dataset generator in this folder.

1. Generate the dataset from scratch (if you want to reproduce the files):

```bash
cd docs/workflow-examples/end-to-end-tutorial
python generate_dataset.py
```

2. Validate and run end-to-end:

```bash
gpurec validate-config \
  --config run.json \
  --check-preprocess \
  --require-mode-default-optimizer \
  --require-cuda-backward-ready

gpurec optimize \
  --config run.json \
  --require-mode-default-optimizer \
  --require-final-check-ok

gpurec summary-info --summary output_gpurec/summary.json --require-final-check-ok
gpurec sample \
  --checkpoint output_gpurec/checkpoints/best.pt \
  --samples 2 \
  --seed 42 \
  --sample-out-dir output_gpurec
```

If your first run is interrupted, resume with `--resume-from`:

```bash
gpurec optimize --config run.json --resume-from output_gpurec/checkpoints/latest.pt
```

The files in this folder are the checked dataset and a ready-to-run config:

- `sp.nwk` (`261` species names; postorder count `261`)
- `families.txt` (two families)
- `gene_map.tsv` (explicit mapping file)
- `g1.nwk`, `g2.nwk` (small gene trees)
- `run.json`
