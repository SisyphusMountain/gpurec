# Notebooks

This directory contains exploratory and end-to-end notebooks for running GPUREC
experiments from Python.

## Files

- `optimize_hogenom_likelihood_end_to_end.ipynb`: end-to-end HOGENOM likelihood
  optimization notebook. It locates input/output paths, checks the native
  preprocessing extension, builds the model, evaluates initial likelihood and
  gradients, runs optimization, plots the trace, inspects fitted event
  probabilities, and saves the final trace and parameters.
