## Overview

This repository provides a modular framework for developing and evaluating jet clustering algorithms in the context of CMS Phase-2 L1 Scouting.

Key features:
- Pluggable clustering backend supporting multiple algorithm families (easily extensible)
- Three-step pipeline separating clustering, studies, and plotting
- Snapshot-based workflow for reproducibility and fast iteration
- Cache-driven analysis layer decoupling heavy computation from visualization
- Comprehensive plotting suite for performance and physics validation

The framework is designed for algorithm R&D, including:
- cone-based approaches (SeededCone variants)
- link-based / CLUE-inspired methods
- reference clustering (anti-kT)

---

## Workflow

The framework follows a three-step pipeline:

### 1. Clustering

- Load input ROOT files (via uproot)
- Run selected clustering algorithms
- Store outputs in a snapshot ROOT file

```
python3 run_clustering.py --config test.py
```

Output:
outputs/<config_tag>/<sample>/snapshot/clustered_events.root

---

### 2. Studies

- Read snapshot
- Perform matching:
  - GEN ↔ RECO
  - RECO ↔ RECO
- Compute:
  - jet-level metrics
  - event-level observables
- Write compact .npz caches

```
python run_studies.py --config test.py
```

Output:
outputs/<config_tag>/<sample>/studies/cache/

---

### 3. Plotting

- Read cached study outputs
- Produce diagnostic and comparison plots

```
python run_plotting.py --config test.py
```

Output:
outputs/<config_tag>/<sample>/plots/

---

## Algorithm Support

Supported algorithm families:

- Reference
  - anti-kT (FastJet)

- Seeded cone
  - Greedy / NMS / weighted variants

- Link-based / CLUE-inspired
  - LinkTree and related variants

All algorithms are registered centrally and controlled entirely via configuration.

---

## Matching and Metrics

The framework provides:

- Greedy one-to-one matching:
  - GEN → RECO
  - RECO → GEN
  - RECO → RECO

- Performance metrics:
  - Efficiency vs pT
  - Jet response and resolution
  - ΔR matching

- Constituent-based comparisons:
  - pT-weighted overlap (IoU-like)
  - unweighted overlap

- Event-level observables:
  - jet multiplicity
  - HT
  - seed statistics

- Algorithm agreement:
  - comparison to reference clustering (e.g. anti-kT)

---

## Configuration

All behavior is controlled via single Python config file:

- Input datasets and branches
- Enabled inputs (e.g. PF, PUPPI)
- Enabled algorithms and parameters
- Matching configuration
- Studies to run
- Plot styles and binning

Configs are resolved automatically from:
configs/<name>.py

Example:
python run_clustering.py --config test.py

Start from:
configs/example_config.py

---

## Dependencies

Core:
- Python ≥ 3.9
- numpy
- awkward
- uproot
- matplotlib

Optional:
- fastjet (for anti-kT reference)
- tqdm (progress bars)

---

## Data and Outputs

- Input ROOT files are not tracked
- data/ is a placeholder for local datasets
- All outputs are written to:
outputs/<config_tag>/

Structure:
snapshot/   → clustering output (ROOT)
studies/    → cached metrics (.npz)
plots/      → figures

---

## Design Principles

- Separation of concerns (clustering vs analysis vs plotting)
- Reproducibility via snapshot + config
- Flexibility via config-driven design
- Transparency over implicit behavior

---

## Citation

If you use this framework in a study or publication, please cite the repository and reference the corresponding analysis or thesis where applicable.

---

## Notes

This is a research-oriented framework, not a production reconstruction package.

The focus is on:
- correctness
- interpretability
- rapid prototyping of new algorithms

Performance optimization (e.g. GPU backends) is considered separately from the analysis layer.
