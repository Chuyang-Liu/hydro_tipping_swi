# hydro_tipping_swi

Python package and reproducibility bundle for the manuscript:

**A Hydrogeologic Tipping Point Controls Climate-Driven Saltwater Intrusion**

This repository contains reusable Python modules, small bundled data products, and example scripts for reproducing key manuscript figures and for applying the workflow to new data.

---

## Overview

This package supports three main tasks:

1. **Pathway analysis and classification**
   - classify saltwater intrusion behavior into transport-limited, lateral-dominated, mixed, and vertical-dominated regimes
   - apply tipping-point based interpretation of permeability controls

2. **Figure reproduction**
   - reproduce bundled manuscript figures from compact CSV/NPZ data products included in this repository

3. **Reusable workflow components**
   - plotting utilities
   - pathway labeling and postprocessing
   - raster-based prediction helpers
   - recharge and permeability preprocessing utilities
   - NOAA sea-level-rise polygon utilities

---

## Repository structure

```text
hydro_tipping_swi/
├── environment.yml
├── README.md
├── LICENSE
├── hydro_tipping_swi/
│   ├── __init__.py
│   ├── plotting.py
│   ├── pathway_labels.py
│   ├── pathway_model.py
│   ├── pathway_predict.py
│   ├── polaris_perm.py
│   ├── recharge.py
│   ├── noaa_download.py
│   ├── noaa_export.py
│   └── io_utils.py
├── examples/
│   └── demo_run.py
└── data/
    ├── repro_bundle_salinity_maps.npz
    ├── df_models_with_salt_budget.csv
    └── summary_with_areas_km2.csv