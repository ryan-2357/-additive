# On additive averaging kernels for finite Markov chains

This repository contains the code used to generate all numerical results in the paper.

---

## Experimental Setup

The experiments focus on the Curie–Weiss model with small dimension `d` (typically for `d = 4,5`), so that brute-force optimisation can be used as a benchmark for our algorithms.

---

## Repository Content

### 1. `MC.py`
Core model builder and shared utilities.

This file contains the main construction of the Curie-Weiss model and the Glauber dynamics, together with some utility functions.

This module is imported by `additive.ipynb`.

---

### 2. `additive.ipynb`
Generates Figures 1–5.

This notebook:
- Computes and compares total variation distances for different averaging samplers 
- Produces magnetisation–mass visualisations for different optimal cuts across different samplers

All plots are fully reproducible from this notebook.
