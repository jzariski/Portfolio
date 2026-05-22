# Telescope Pointing Offset Pipeline

This directory contains a small command-line pipeline for learning telescope
pointing corrections from observation logs. It parses raw logs, trains two
XGBoost regressors, optionally runs feature sensitivity analysis, and predicts
the correction for a new acquisition.

The model is autoregressive: each prediction uses the previous acquisition's RA
and Dec offset as input features. During training those lagged features come
from the previous row's true `obs - solv` error. During live prediction they
come from the previous prediction saved in the log file.

## Files

```text
load_out.py   Parse raw observation text into data/data.h5
booster.py    Train RA and Dec offset models and save diagnostics
predict.py    Predict one acquisition using saved models
Sobol.py      Optional Sobol sensitivity analysis for trained models
Dockerfile    Container recipe with the required Python packages
data/         Raw input logs and generated HDF5 data
models/       Generated model_ra.json and model_dec.json
logs/         Prediction and Sobol output logs
plots/        Generated diagnostic plots
```

## Input Data

Raw observation files should be whitespace-delimited with these six columns:

```text
DATE-OBS LST OBS-RA OBS-DEC CRVAL1 CRVAL2
```

Column meanings:

```text
DATE-OBS  UTC timestamp, for example 2026-01-04T01:40:41.408
LST       Local sidereal time as HH:MM:SS.sss
OBS-RA    Commanded/observed RA as HH:MM:SS.sss
OBS-DEC   Commanded/observed Dec as +/-DD:MM:SS.sss
CRVAL1    Solved RA in degrees
CRVAL2    Solved Dec in degrees
```

`data/data.dat` is an example input file in this format.

## Install

Using a local Python environment:

```bash
python -m pip install numpy matplotlib h5py astropy SALib xgboost scikit-learn
```

Or build the included container:

```bash
docker build -t telescope-api .
docker run --rm -it -v "$PWD":/app telescope-api
```

## Workflow

1. Convert the raw log into HDF5:

```bash
python load_out.py --input data/data.dat
```

Useful options:

```text
--dt 60       Keep only rows more than 60 seconds after the previous row
--toCIRS      Transform solved RA/Dec from ICRS into apparent CIRS coordinates
--sim 3       Keep only the first 3 consecutive similar pointings
--eps 0.1     Similarity threshold in degrees for --sim
```

This writes `data/data.h5`.

2. Train the models:

```bash
python booster.py
```

This writes:

```text
models/model_ra.json
models/model_dec.json
plots/cdf.png
```

3. Predict a new acquisition:

```bash
python predict.py \
  --year 2026 \
  --month 1 \
  --day 4 \
  --hour 1 \
  --minute 40 \
  --second 41 \
  --lst-hours 0.0567 \
  --obs-ra-deg 119.2954167 \
  --obs-dec-deg 78.4702361 \
  --lat-deg 31.9583 \
  --lon-deg -111.5986 \
  --elevation-m 2400 \
  --models-dir models \
  --log-file logs/prediction_log.txt
```

`predict.py` prints the predicted RA/Dec correction, the corrected equatorial
coordinates, and the corresponding Alt/Az values. It appends the prediction to
the log so the next call has previous-error features available.

4. Optional sensitivity analysis:

```bash
python Sobol.py --n-base 4096
```

This writes `logs/sobol_output.txt`. Larger `--n-base` values give more stable
Sobol estimates but take longer.

## Feature Order

All model scripts use this exact input order:

```text
year
month
day
hour
minute
second
lst_hours
obs_ra_deg
obs_dec_deg
previous_acq_error_ra
previous_acq_error_dec
```

The target labels are:

```text
ra_offset  = obs_ra_deg  - solv_ra_deg
dec_offset = obs_dec_deg - solv_dec_deg
```

The predicted corrected coordinates are therefore:

```text
predicted_solv_ra  = obs_ra_deg  - predicted_ra_offset
predicted_solv_dec = obs_dec_deg - predicted_dec_offset
```

## Notes

- Keep training and prediction coordinate frames consistent. If you train with
  `--toCIRS`, feed prediction coordinates in the matching apparent frame.
- The first prediction in a fresh log uses `0.0, 0.0` for previous-error
  features because no past prediction exists yet.
- `booster.py` splits chronologically, so later observations are never used to
  train earlier predictions.
