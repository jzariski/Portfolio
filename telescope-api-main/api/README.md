# Telescope API Pipeline

A command-line toolchain for telescope pointing offset prediction and model exploration.
This readme covers the `api/` folder, which includes data parsing, training, sensitivity analysis, and prediction.

## Project structure

```
.
├── load_out.py          # parse raw log text into usable HDF5 training data
├── booster.py           # train models and save model artifacts + diagnostics
├── Sobol.py             # run Sobol sensitivity analysis
├── predict.py           # predict observations from saved models
├── Dockerfile           # container recipe for the API service
├── data/                # raw input and generated HDF5
├── models/              # saved model artifacts and JSON models
├── logs/                # prediction logs and Sobol outputs
└── plots/               # diagnostic plots and CDF visualizations
```

## Training data format

The training file should be whitespace-delimited and begin with a header row.
The first six columns must be:

```
DATE-OBS     LST     OBS-RA     OBS-DEC     CRVAL1     CRVAL2
```

- `DATE-OBS`: ISO date/time, for example `2025-05-09T04:26:12.247Z`
- `LST`: sidereal time as `HH:MM:SS[.sss]`
- `OBS-RA`: observed RA in hours format
- `OBS-DEC`: observed Dec in degrees format
- `CRVAL1`: solution RA in degrees
- `CRVAL2`: solution Dec in degrees

## Quick start

```bash
python load_out.py --input data/your_observations.txt
python booster.py
python predict.py --year 2025 --month 5 --day 9 --hour 4 --minute 26 --second 0 \
  --lst-hours 5.25 --OBS-ra-deg 123.456 --OBS-dec-deg -20.000 \
  --lat-deg 31.9583 --lon-deg -111.5986
```

## Command-line options

- `--dt`: minimum spacing in seconds between retained rows
- `--toCIRS`: transform solution coordinates to CIRS frame before training
- `--sim`: retain only the first `K` similar consecutive acquisitions
- `--eps`: similarity threshold for the `--sim` filter

## Recommended workflow

1. Run `load_out.py` to create `data/data.h5` from raw logs.
2. Run `booster.py` to train models and save artifacts.
3. Optionally run `Sobol.py` for feature sensitivity insight.
4. Run `predict.py` to generate predictions from the trained models.

## Notes for reviewers

- This repository separates parsing, modeling, analysis, and inference cleanly.
- It is built to demonstrate applied telescope data processing and model deployment.
- Output folders are kept isolated so artifacts are easy to inspect.
