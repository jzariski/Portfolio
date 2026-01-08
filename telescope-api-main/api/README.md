# LCO API

## LCO Telescope API — Docker Quickstart

Predicts telescope pointing offsets from a LCO historical observation log via an API.  

### Requirements

- Docker Desktop (macOS/Windows) or Docker Engine (Linux)

### Project layout

```
.
├── load_out.py          # Step 1: parse raw logs → HDF5
├── booster.py           # Step 2: train XGB models, save models + plots
├── Sobol.py             # Step 3: Sobol sensitivity analysis (SALib)
├── predict.py           # Step 4: repeated predictions using trained models
├── Dockerfile
├── data/                # raw input logs + generated data.h5
├── models/              # model_ra.json, model_dec.json
├── logs/                # Sobol output, prediction logs
└── plots/               # CDF and other diagnostic plots
```

### Input data format (training)

Your training file is a whitespace-delimited text file whose **first six columns (in this exact order)** are:

```
DATE-OBS     LST     TPT-RA     TPT-DEC     CRVAL1     CRVAL2
```

- **DATE-OBS**: ISO date/time (e.g. `2025-05-09T04:26:12.247Z` or without `Z`)
- **LST**: sidereal time (`HH:MM:SS[.sss]`)
- **TPT-RA**: pointing RA (`HH:MM:SS[.sss]`) — hours, no wrapping performed
- **TPT-DEC**: pointing Dec (`±DD:MM:SS[.ss]`)
- **CRVAL1**: WCS RA in **degrees**
- **CRVAL2**: WCS Dec in **degrees**


---



# 1) Open Docker and build the image

```bash
open -a "Docker"
```

```bash
docker build -t lco-api .
```

Important: If used in the past it's a good idea to empty the logs and plots folder

# 2) Keeps outputs on the host machine

```bash
docker run -it --rm -v "$PWD":/app lco-api
```

# 3.1) Format Data

This takes your text file 'file_path' and transforms it into a useable h5 file stored in the data directory

```bash
python load_out --input 'file_path'
```
Other Arguments
- --dt: Default 0.0. Minimum time difference between successive points kept in training set.
- --toCIRS: Default True. Converts WCS to CIRS frame (assuming ICRS originally).
- --sim: Default 0. Number of consecutive similar acquisition points to keep in traiing data.
- --eps: Default 0.1. Defines similarity if sim > 0.


# 3.2) Trains an RA/Dec Model

Right now we've chosen the specs of the model for you. We're adding in ways that the user can change it soon
This model is trained to predict the offset between TPT RA/DEC and WCS (CRVAL) RA/DEC

```bash
python booster.py
```

# 3.3) Optional Sobol Analysis (Output in logs)

Optional variance analysis for feature engineering. Sobol analysis output in logs directory


```bash
python Sobol.py
```

# 4) Predict

Used to predict offset between TPT and WCS. Uses former predicted offsets as features, keeping track in 
prediction_log.txt in the logs directory. Most recent entry is what is used, to restart, delete the txt file.

```bash
python predict.py \
  --year int \
  --month int \
  --day int \
  --hour int \
  --minute int \
  --second int \
  --lst-hours float \
  --tpt-ra-deg float \
  --tpt-dec-deg float \
  --lat-deg float \
  --lon-deg float \
  --elevation-m float \
  --models-dir (Default) models \
  --log-file (Default) prediction_log.txt
```
