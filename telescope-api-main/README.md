# Telescope API Container

This project contains two complementary implementations of a telescope pointing offset pipeline:

- `api/` — a Docker-ready, command-line based training and prediction workflow.
- `api-flask/` — a Flask web API wrapper that exposes the pipeline via HTTP endpoints.

The goal is to demonstrate both standalone Python tooling and a lightweight API interface for telescope prediction workflows.

## Contents

- `api/` — data conversion, model training, Sobol analysis, and prediction scripts.
- `api-flask/` — Flask server with `/train` and `/predict` routes.
- `Dockerfile` — container entrypoint for the Flask API version.

## Usage

The `api` folder is designed for command-line use and batch processing.
The `api-flask` folder is designed for deployment as a containerized service, with an HTTP interface for training and inference.

## Why this matters

This repository shows the ability to take an ML prediction pipeline and wrap it in both a reusable CLI and a deployable API surface. It is intended for telescope operations teams that want a scripted workflow plus a service endpoint.

