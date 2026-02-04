## Overview

This repository provides:
- A pluggable jet clustering backend supporting multiple categories of algorithms (extentable)
- A config-driven processing pipeline to run clustering, matching, and metric extraction
- A cache-based workflow separating heavy processing from fast plotting
- A comprehensive plotting suite for efficiency, response, purity/fake rates, event-level observables, and algorithm agreement

The framework is intended for research and development of jet clustering algorithms, including classical cone-based approaches, simplified CLUE-like linking methods, and reference anti-kT clustering.

## Repository Structure

JetClustering/
├── src/
│   ├── clustering_algorithms.py   # Jet clustering implementations + registry
│   ├── utils.py                   # Matching, overlap metrics, IO helpers
│   └── plotting_utils.py          # Plotting routines
│
├── configs/
│   ├── example_config.py          # Minimal working configuration
│   └── *.py                       # User-defined configs (ignored by git)
│
├── data/
│   └── .gitignore                 # Placeholder for input ROOT files
│
├── outputs/                       # Generated caches and plots (ignored)
│
├── run_processing.py              # Processing + cache production
├── run_plots.py                   # Plotting from cached outputs
├── LICENSE
└── README.md

## Workflow

The framework follows a two-step workflow.

### Processing

- Load ROOT files via uproot
- Run selected clustering algorithms
- Perform GEN↔RECO and RECO↔RECO matching
- Compute per-jet and per-event observables
- Write compact .npz cache files

python run_processing.py --config configs/example_config.py

### Plotting

- Read cached outputs
- Produce diagnostic plots

python run_plots.py --config configs/example_config.py

## Supported Algorithm Families

Reference:
- anti-kT (FastJet)

Seeded cone

Link-based / CLUE-inspired

All algorithms are registered via a central registry and can be enabled or disabled purely through configuration.

## Matching and Metrics

The framework includes:
- Greedy one-to-one GEN→RECO, RECO→GEN, and RECO→RECO matching
- Jet response and resolution metrics
- pT-weighted and unweighted constituent overlap (IoU)
- Event-level observables: jet multiplicities, HT, seed statistics
- Algorithm agreement with a reference clustering (anti-kT).

## Configuration

All behavior is controlled via Python configuration files:
- Input datasets and branches
- Enabled inputs (PF, PUPPI, etc.)
- Enabled algorithms and parameters
- Matching thresholds
- Studies to run
- Plot styles and binning

Start from configs/example_config.py and extend as needed.

## Dependencies

Core dependencies:
- Python ≥ 3.9
- numpy
- awkward
- uproot
- matplotlib

Optional:
- fastjet (for anti-kT)
- tqdm (progress bars)

## Data Handling

- Input ROOT files are not tracked
- The data/ directory is a placeholder for local datasets
- All outputs are written to outputs/<config_tag>/

## License

This project is licensed under the Apache License 2.0.

## Citation

If you use this framework in a study or publication, please cite the repository and reference the relevant analysis or thesis where appropriate.

## Notes

This is a research framework, not a production reconstruction package.
Correctness, determinism, and transparency are prioritized over raw performance.

Contributions and algorithmic extensions are welcome.
