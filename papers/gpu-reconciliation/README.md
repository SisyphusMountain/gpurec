# Differentiable DTL reconciliation paper

This is the canonical manuscript formerly maintained in
`kernel-bench/paper/gpurec_paper.tex`.

- Source: `main.tex`
- Publication-input figures: `figures/`
- Small saved inputs used to redraw figures: `reproduction/`
- Paper-specific redraw scripts: `scripts/`
- HOGENOM CPU/GPU benchmark: `../../benchmarks/hogenom-cpu-vs-gpu/`
- Model-selection and curvature drivers: `../../experiments/sanderson_cv/`
- Receiver-robustness drivers: `../../experiments/receiver_robustness/`

## Build

```sh
latexmk -pdf main.tex
```

The generated `main.pdf` and LaTeX intermediates are ignored. The six figure
PDFs are tracked because they are manuscript inputs.

## Provenance

The imported source and figures came from local `kernel-bench` branch `paper`
at commit `905d88a6bd3669bc0b193b4fded726e67b973de6`. The working PDF and saved
profile input were preserved on its local rescue commit `4f458af` before
migration. A complete all-ref Git bundle is retained in
`archive/storage/repository-bundles/`.

## Figure map

| Figure | Generator/evidence |
|---|---|
| `fig_speed.pdf` | `../../benchmarks/hogenom-cpu-vs-gpu/plots/make_fig_speed.py` reading `results/headline.csv` |
| `fig_cv.pdf` | `../../experiments/sanderson_cv/plot_fig_cv.py` reading the saved CV grids under `reproduction/cv/` |
| `fig_profile.pdf` | `scripts/replot_profile_sym.py` reading `reproduction/profile/fig_profile.pt` |
| `fig_s53_spectrum.pdf` | `../../experiments/sanderson_cv/fisher_information_s53.py` |
| `fig_s53_dl_confounding.pdf` | same Fisher-information driver |
| `fig_s53_se.pdf` | same Fisher-information driver |
