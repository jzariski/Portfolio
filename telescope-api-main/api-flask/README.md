# Telescope API (FLASK VERSION)

## Telescope API — Docker Quickstart

Predicts telescope pointing offsets from a Telescope historical observation log via an API. Flask Version  

### Requirements

- Docker Desktop (macOS/Windows) or Docker Engine (Linux)
- Curl

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



1) Open Docker and build the image

```bash
open -a "Docker"
```

```bash
docker build -t Telescope-api-flask .

docker run --rm -d \
  --name Telescope-api-flask \
  -p 5000:5000 \
  -v "$PWD:/app" \
  Telescope-api-flask
```

Important: If used in the past it's a good idea to empty the logs and plots folder

2) Example call for training the model
```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "data/2025_05_09.txt",
    "min_dt": 60.0,
    "to_cirs": true,
    "sim_k": 0,
    "eps": 0.1
  }'
```

OR to get a clean output
```bash
curl -X POST "http://localhost:5000/train?format=text" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "data/2025_05_09.txt",
    "min_dt": 60.0,
    "to_cirs": true,
    "sim_k": 0,
    "eps": 0.1
  }'
```







Arguments
- -- input_file: Path to the txt file containing historical telemetry.
- --min_dt: Default 0.0. Minimum time difference between successive points kept in training set.
- --to_cirs: Default True. Converts WCS to CIRS frame (assuming ICRS originally).
- --sim_k: Default 0. Number of consecutive similar acquisition points to keep in traiing data.
- --eps: Default 0.1. Defines similarity if sim > 0.


3) Example call for predicting with the model
Used to predict offset between TPT and WCS. Uses former predicted offsets as features, keeping track in 
prediction_log.txt in the logs directory. Most recent entry is what is used, to restart, delete the txt file.
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2025,
    "month": 12,
    "day": 15,
    "hour": 8,
    "minute": 30,
    "second": 0,
    "lst_hours": 5.75,
    "tpt_ra_deg": 123.456,
    "tpt_dec_deg": -20.0,
    "lat-deg": 31.9583,
    "lon-deg": -111.5986,
    "elevation-m": 2400.0,
    "log-file": "logs/prediction_log.txt"
  }'
```


OR to get a clean output

```bash
curl -X POST "http://localhost:5000/predict?format=text" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2025,
    "month": 12,
    "day": 15,
    "hour": 8,
    "minute": 30,
    "second": 0,
    "lst_hours": 5.75,
    "tpt_ra_deg": 123.456,
    "tpt_dec_deg": -20.0,
    "lat-deg": 31.9583,
    "lon-deg": -111.5986,
    "elevation-m": 2400.0,
    "log-file": "logs/prediction_log.txt"
  }'
```

4) Kill the container
```bash
docker rm -f Telescope-api-flask
```


