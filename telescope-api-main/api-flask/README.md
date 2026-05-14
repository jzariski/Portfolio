# Flask API for Telescope Prediction

A lightweight HTTP wrapper around the telescope pipeline.
This folder contains a Docker-ready Flask service with `/train` and `/predict` endpoints.

## Requirements

- Docker Desktop or Docker Engine
- `curl` for local API testing

## Input data format

The same whitespace-delimited training format used by the core pipeline applies here.
The first six columns must be:

```
DATE-OBS     LST     obs-RA     obs-DEC     CRVAL1     CRVAL2
```

- `DATE-OBS`: ISO date/time, for example `2025-05-09T04:26:12.247Z`
- `LST`: sidereal time as `HH:MM:SS[.sss]`
- `obs-RA`: observed RA in hours format
- `obs-DEC`: observed Dec in degrees format
- `CRVAL1`: solution RA in degrees
- `CRVAL2`: solution Dec in degrees

## Build and run

```bash
open -a "Docker"
docker build -t Telescope-api-flask .
docker run --rm -d \
  --name Telescope-api-flask \
  -p 5000:5000 \
  -v "$PWD:/app" \
  Telescope-api-flask
```

> If the container has been run before, clear the `logs/` and `plots/` folders for a clean start.

## Train endpoint example

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

Use `?format=text` for a human-readable summary instead of JSON.

## Predict endpoint example

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
    "obs_ra_deg": 123.456,
    "obs_dec_deg": -20.0,
    "lat-deg": 31.9583,
    "lon-deg": -111.5986,
    "elevation-m": 2400.0,
    "log-file": "logs/prediction_log.txt"
  }'
```

Use `?format=text` for plain text output from the same request.

## Notes on training parameters

- `input_file`: path to the historical telemetry file
- `min_dt`: minimum allowed time difference between successive samples
- `to_cirs`: convert solution coordinates to CIRS frame
- `sim_k`: number of similar consecutive observation points to retain
- `eps`: similarity threshold for the `sim_k` filter

## Stop and remove the container

```bash
docker rm -f Telescope-api-flask
```
