#!/usr/bin/env python3
"""
app.py

Tiny Flask API on top of the existing pipeline.

Endpoints:
  POST /train   -> run load_out.py then booster.py (with JSON args)
  POST /predict -> call predict.py with JSON args, return parsed results

Both endpoints support an optional query parameter:
  ?format=json  (default) → JSON response
  ?format=text           → human-readable text summary
"""

import os
import sys
import subprocess
from flask import Flask, request, jsonify, Response

app = Flask(__name__)


# ----------------------------------------------------------------------------- 
# Helpers
# -----------------------------------------------------------------------------

def run_script(cmd, cwd=None, env=None):
    """
    Run a Python script as a subprocess and capture stdout/stderr.

    Args:
        cmd: list[str], e.g. [sys.executable, "load_out.py"]
        cwd: working directory (default: None)
        env: environment dict (default: inherit from parent)

    Returns:
        (returncode, stdout, stderr)
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def parse_predict_output(stdout: str):
    """
    Parse the stdout from predict.py and pull out key numerical values.

    Designed to work with a predict.py that prints lines like:

      === Prediction ===
      TPT RA (deg):  ...
      TPT DEC (deg): ...
      Predicted RA offset (deg):  ...
      Predicted DEC offset (deg): ...

    Optionally, if you print Alt/Az lines such as:

      TPT ALT (deg):  ...
      TPT AZ  (deg):  ...
      Predicted ALT offset (deg): ...
      Predicted AZ offset (deg):  ...

    those will also be parsed.
    """
    vals = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        def grab(prefix, key):
            if line.startswith(prefix):
                try:
                    number_str = line.split(":", 1)[-1].strip().split()[-1]
                    vals[key] = float(number_str)
                except Exception:
                    pass

        grab("Predicted RA offset (deg):", "ra_offset_deg")
        grab("Predicted DEC offset (deg):", "dec_offset_deg")


        # Optional Alt/Az lines
        grab("TPT ALT (deg):", "tpt_alt_deg")
        grab("TPT AZ", "tpt_az_deg")  # matches "TPT AZ  (deg):"
        grab("Predicted ALT offset (deg):", "alt_offset_deg")
        grab("Predicted AZ offset (deg):", "az_offset_deg")

    return vals


def text_response(body: str, status: int = 200):
    """Return plain-text HTTP response."""
    return Response(body + "\n", status=status, mimetype="text/plain")


# ----------------------------------------------------------------------------- 
# Routes
# -----------------------------------------------------------------------------

@app.route("/train", methods=["POST"])
def train():
    """
    Run the full training pipeline:

      1) load_out.py
      2) booster.py

    JSON body (all except input_file are optional):

    {
      "input_file": "data/2025_05_09.txt",  # REQUIRED
      "min_dt": 60.0,       # seconds, default 0.0 (no time filter)
      "to_cirs": true,      # default false
      "sim_k": 0,           # default 0 (no similarity filter)
      "eps": 0.1            # similarity threshold (only if sim_k > 0)
    }

    Query parameter:
      ?format=json  (default)
      ?format=text
    """
    fmt = request.args.get("format", "json")
    data = request.get_json(force=True) or {}

    if "input_file" not in data:
        msg = "Missing required key 'input_file'"
        if fmt == "text":
            return text_response(f"=== Training FAILED ===\n{msg}", status=400)
        return jsonify({"status": "error", "message": msg}), 400

    input_file = data["input_file"]
    min_dt = float(data.get("min_dt", 0.0))
    to_cirs = bool(data.get("to_cirs", False))
    sim_k = int(data.get("sim_k", 0))
    eps = float(data.get("eps", 0.1))

    # ---- Step 1: run load_out.py with CLI args ----
    cmd1 = [sys.executable, "load_out.py", "--input", input_file]

    if min_dt > 0:
        cmd1 += ["--dt", str(min_dt)]
    if to_cirs:
        cmd1.append("--toCIRS")
    if sim_k > 0:
        cmd1 += ["--sim", str(sim_k), "--eps", str(eps)]

    code1, out1, err1 = run_script(cmd1, cwd="/app")
    if code1 != 0:
        payload = {
            "status": "error",
            "stage": "load_out",
            "input_file": input_file,
            "stdout": out1,
            "stderr": err1,
        }
        if fmt == "text":
            lines = [
                "=== Training FAILED at stage: load_out ===",
                f"Input file: {input_file}",
                "",
                "--- load_out.py STDOUT ---",
                out1.strip(),
                "",
                "--- load_out.py STDERR ---",
                err1.strip(),
            ]
            return text_response("\n".join(lines), status=500)
        return jsonify(payload), 500

    # ---- Step 2: run booster.py (reads data/data.h5) ----
    cmd2 = [sys.executable, "booster.py"]
    code2, out2, err2 = run_script(cmd2, cwd="/app")
    output_h5 = "data/data.h5"  # current load_out.py always writes here

    if code2 != 0:
        payload = {
            "status": "error",
            "stage": "booster",
            "input_file": input_file,
            "output_h5": output_h5,
            "stdout": out2,
            "stderr": err2,
        }
        if fmt == "text":
            lines = [
                "=== Training FAILED at stage: booster ===",
                f"Input file: {input_file}",
                f"Output HDF5: {output_h5}",
                "",
                "--- booster.py STDOUT ---",
                out2.strip(),
                "",
                "--- booster.py STDERR ---",
                err2.strip(),
            ]
            return text_response("\n".join(lines), status=500)
        return jsonify(payload), 500

    # Success payload
    payload = {
        "status": "ok",
        "message": "Training completed successfully.",
        "input_file": input_file,
        "output_h5": output_h5,
        "load_out_stdout": out1,
        "booster_stdout": out2,
    }

    if fmt == "text":
        lines = [
            "=== Training COMPLETED ===",
            f"Input file: {input_file}",
            f"Output HDF5: {output_h5}",
            "",
            "--- load_out.py STDOUT (truncated if long) ---",
            out1.strip(),
            "",
            "--- booster.py STDOUT (truncated if long) ---",
            out2.strip(),
        ]
        return text_response("\n".join(lines))

    return jsonify(payload)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Call predict.py using JSON parameters from the request body.
    Returns parsed numeric results as structured JSON or pretty text.

    Expected JSON body (hyphen OR underscore keys accepted where noted):

    {
      "year": 2025,
      "month": 12,
      "day": 15,
      "hour": 8,
      "minute": 30,
      "second": 0,
      "lst_hours": 5.75,

      "tpt_ra_deg": 123.456,
      "tpt_dec_deg": -20.0,

      "lat_deg": 31.9583,    # or "lat-deg"
      "lon_deg": -111.5986,  # or "lon-deg"

      "elevation_m": 2400.0, # or "elevation-m" (optional)
      "models_dir": "models",
      "log_file": "logs/prediction_log.txt"  # or "log-file"
    }

    Query parameter:
      ?format=json  (default)
      ?format=text
    """
    fmt = request.args.get("format", "json")
    data = request.get_json(force=True) or {}

    # Basic required scalar fields
    scalar_required = [
        "year", "month", "day",
        "hour", "minute", "second",
        "lst_hours",
        "tpt_ra_deg", "tpt_dec_deg",
    ]
    missing_scalar = [k for k in scalar_required if k not in data]
    if missing_scalar:
        msg = f"Missing required keys: {missing_scalar}"
        if fmt == "text":
            return text_response(f"=== Prediction FAILED ===\n{msg}", status=400)
        return jsonify({"status": "error", "message": msg}), 400

    # Handle lat/lon with both underscore and hyphen spellings
    lat = data.get("lat_deg", data.get("lat-deg"))
    lon = data.get("lon_deg", data.get("lon-deg"))

    missing_latlon = []
    if lat is None:
        missing_latlon.append("lat_deg or lat-deg")
    if lon is None:
        missing_latlon.append("lon_deg or lon-deg")

    if missing_latlon:
        msg = f"Missing required keys: {missing_latlon}"
        if fmt == "text":
            return text_response(f"=== Prediction FAILED ===\n{msg}", status=400)
        return jsonify({"status": "error", "message": msg}), 400

    # Optional fields with aliases
    elevation = data.get("elevation_m", data.get("elevation-m", 0.0))
    models_dir = data.get("models_dir", "models")
    log_file = data.get("log_file", data.get("log-file", "prediction_log.txt"))

    cmd = [
        sys.executable, "predict.py",
        "--year", str(data["year"]),
        "--month", str(data["month"]),
        "--day", str(data["day"]),
        "--hour", str(data["hour"]),
        "--minute", str(data["minute"]),
        "--second", str(data["second"]),
        "--lst-hours", str(data["lst_hours"]),
        "--tpt-ra-deg", str(data["tpt_ra_deg"]),
        "--tpt-dec-deg", str(data["tpt_dec_deg"]),
        "--lat-deg", str(lat),
        "--lon-deg", str(lon),
        "--elevation-m", str(elevation),
        "--models-dir", models_dir,
        "--log-file", log_file,
    ]

    code, out, err = run_script(cmd, cwd="/app")
    if code != 0:
        payload = {
            "status": "error",
            "message": "predict.py failed",
            "stdout": out,
            "stderr": err,
        }
        if fmt == "text":
            lines = [
                "=== Prediction FAILED (predict.py error) ===",
                "",
                "--- STDOUT ---",
                out.strip(),
                "",
                "--- STDERR ---",
                err.strip(),
            ]
            return text_response("\n".join(lines), status=500)
        return jsonify(payload), 500

    parsed = parse_predict_output(out)

    # Build a more structured / readable JSON payload
    offsets_equatorial = {
        "ra_deg": parsed.get("ra_offset_deg"),
        "dec_deg": parsed.get("dec_offset_deg"),
    }


    tpt_horizon = {
        "alt_deg": parsed.get("tpt_alt_deg"),
        "az_deg": parsed.get("tpt_az_deg"),
    }

    offsets_horizon = {
        "alt_deg": parsed.get("alt_offset_deg"),
        "az_deg": parsed.get("az_offset_deg"),
    }


    payload = {
        "status": "ok",
        "input": {
            "year": data["year"],
            "month": data["month"],
            "day": data["day"],
            "hour": data["hour"],
            "minute": data["minute"],
            "second": data["second"],
            "lst_hours": data["lst_hours"],
            "tpt_ra_deg": data["tpt_ra_deg"],
            "tpt_dec_deg": data["tpt_dec_deg"],
            "lat_deg": float(lat),
            "lon_deg": float(lon),
            "elevation_m": float(elevation),
        },
        "prediction": {
            "offsets_equatorial": offsets_equatorial,
            "wcs_equatorial": wcs_equatorial,
            "tpt_horizon": tpt_horizon,
            "offsets_horizon": offsets_horizon,
            "wcs_horizon": wcs_horizon,
        },
        "log_file": log_file,
        "raw_stdout": out,
    }

    if fmt == "text":
        # Build a nice multiline summary
        inp = payload["input"]
        pred = payload["prediction"]

        lines = [
            "=== Prediction SUMMARY ===",
            "",
            "Input:",
            f"  UTC time      : {inp['year']:04d}-{inp['month']:02d}-{inp['day']:02d} "
            f"{inp['hour']:02d}:{inp['minute']:02d}:{inp['second']:02d}",
            f"  LST (hours)   : {inp['lst_hours']}",
            f"  TPT RA/Dec    : {inp['tpt_ra_deg']} deg, {inp['tpt_dec_deg']} deg",
            f"  Site lat/lon  : {inp['lat_deg']} deg, {inp['lon_deg']} deg",
            f"  Elevation     : {inp['elevation_m']} m",
            "",
            "Equatorial offsets (TPT - WCS):",
            f"  RA offset     : {offsets_equatorial['ra_deg']}",
            f"  Dec offset    : {offsets_equatorial['dec_deg']}",
            "",
        ]

        # Only print horizon stuff if available
        if any(v is not None for v in tpt_horizon.values()):
            lines += [
                "",
                "TPT horizon coordinates:",
                f"  Alt           : {tpt_horizon['alt_deg']}",
                f"  Az            : {tpt_horizon['az_deg']}",
            ]
        if any(v is not None for v in offsets_horizon.values()):
            lines += [
                "",
                "Horizon offsets (WCS - TPT):",
                f"  Alt offset    : {offsets_horizon['alt_deg']}",
                f"  Az offset     : {offsets_horizon['az_deg']}",
            ]
        if any(v is not None for v in wcs_horizon.values()):
            lines += [
                "",
                "Suggested WCS horizon coordinates:",
                f"  Alt           : {wcs_horizon['alt_deg']}",
                f"  Az            : {wcs_horizon['az_deg']}",
            ]

        lines += [
            "",
            f"Prediction log file: {log_file}",
        ]

        return text_response("\n".join(lines))

    return jsonify(payload)


if __name__ == "__main__":
    # Listen on 0.0.0.0 so Docker port mapping works
    app.run(host="0.0.0.0", port=5000)
