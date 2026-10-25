# Causal Deconfounding and Transport for Vocal Parkinson's Disease Detection

## About

Code for a dissertation project on causal deconfounding and cross-corpus
transport for Parkinson's disease detection from voice. It compares
confounded and deconfounded classifiers on NeuroVoz and ParkCeleb, and
transports causal-bootstrapping weights from NeuroVoz to ParkCeleb
disease-duration intervals.

## Requirements

- Python 3.12
- Core packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `openpyxl`, `tqdm`, `matplotlib`
- Optional extraction/SSL arms: `xgboost`, `torch`, `soundfile`, `librosa`
- `ffmpeg` on `PATH` for NISQA

```
pip install numpy pandas scipy scikit-learn openpyxl tqdm matplotlib
```

## Data

Datasets are not included in the repository. Place the inputs under
`data/` (gitignored), using the layout expected by the scripts:

```
data/ParkCeleb/
data/parkceleb_extracted/
data/neurovoz/
data/grbas/
data/updrs/
data/parkceleb_ages.csv
data/parkceleb_upload_years.csv
```

Run the scripts in `extraction/` to build any derived features.

## Run

```
python main.py
```

Optional arguments: `--pc-reps`, `--nv-reps`, `--xfer-reps`,
`--cb-repeats`, `--seed`, and `--transport-only`. Results are written to
`outputs/` (gitignored).

## Distinction

This project achieved a Distinction.
