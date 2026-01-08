#!/usr/bin/env python3
"""
predict.py

Use trained XGBoost models (saved as JSON) to make repeated predictions.

Features:
  year, month, day,
  hour, minute, second,
  lst_hours,
  obs_ra_deg, obs_dec_deg,
  previous_acq_error_ra, previous_acq_error_dec

The "previous_*" features are taken from the most recent prediction
stored in a plain text log file. If no previous prediction exists,
these are initialized to zero.

Log format (whitespace-separated, one prediction per line):
  timestamp_iso  SOLV_ra_deg  SOLV_dec_deg  ra_offset_pred  dec_offset_pred  obs_ra  obs_dec
"""

import argparse
import os
from datetime import datetime

import numpy as np
import xgboost as xgb
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u


DEFAULT_MODELS_DIR = "models"
DEFAULT_LOG_FILE = "prediction_log.txt"


def read_previous_from_log(log_path: str):
    """
    Read the most recent prediction from the log.

    Returns:
        previous_acq_error_ra,
        previous_acq_error_dec

    If the log does not exist or cannot be parsed, returns zeros.
    """
    if not os.path.exists(log_path):
        return 0.0, 0.0

    try:
        with open(log_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            return 0.0, 0.0

        last = lines[-1]
        parts = last.split()

        # Expected format:
        # 0: timestamp_iso
        # 1: SOLV_ra_deg
        # 2: SOLV_dec_deg
        # 3: ra_offset_pred
        # 4: dec_offset_pred
        # 5: obs_ra
        # 6: obs_dec

        ra_offset_pred = float(parts[3])
        dec_offset_pred = float(parts[4])

        previous_acq_error_ra = ra_offset_pred
        previous_acq_error_dec = dec_offset_pred

        return previous_acq_error_ra, previous_acq_error_dec

    except Exception:
        # If anything goes wrong, fall back to zeros
        return 0.0, 0.0


def append_to_log(
    log_path: str,
    SOLV_ra_deg: float,
    SOLV_dec_deg: float,
    ra_offset_pred: float,
    dec_offset_pred: float,
    obs_ra: float,
    obs_dec: float,
):
    """
    Append a single prediction to the log file.

    Line format:
      timestamp_iso  SOLV_ra_deg  SOLV_dec_deg  ra_offset_pred  dec_offset_pred  obs_ra  obs_dec
    """
    timestamp = datetime.utcnow().isoformat()
    line = (
        f"{timestamp} "
        f"{SOLV_ra_deg:.10f} {SOLV_dec_deg:.10f} "
        f"{ra_offset_pred:.10f} {dec_offset_pred:.10f} "
        f"{obs_ra:.10f} {obs_dec:.10f}\n"
    )
    with open(log_path, "a") as f:
        f.write(line)


def build_feature_vector(args, prev_vals):
    """
    Build the feature vector in the same order used during training.

    Args:
        args: parsed command-line args
        prev_vals: tuple of (prev_error_ra, prev_error_dec)

    Returns:
        X: np.ndarray of shape (1, n_features)
    """
    prev_error_ra, prev_error_dec = prev_vals

    features = [
        float(args.year),
        float(args.month),
        float(args.day),
        float(args.hour),
        float(args.minute),
        float(args.second),
        float(args.lst_hours),
        float(args.obs_ra_deg),
        float(args.obs_dec_deg),
        float(prev_error_ra),
        float(prev_error_dec),
    ]

    X = np.array([features], dtype=float)
    return X


def compute_horizon_offsets(args, SOLV_ra_deg: float, SOLV_dec_deg: float):
    """
    Compute horizon-coordinate (Alt/Az) info for obs and SOLV, and their offsets.

    Returns:
        alt_offset_deg, az_offset_deg,
        obs_alt_deg, obs_az_deg,
        SOLV_alt_deg, SOLV_az_deg

    alt_offset_deg = SOLV_alt - obs_alt
    az_offset_deg  = SOLV_az  - obs_az
    """
    # Telescope location
    location = EarthLocation(
        lat=args.lat_deg * u.deg,
        lon=args.lon_deg * u.deg,
        height=args.elevation_m * u.m,
    )

    # Observation time (UTC)
    obstime = Time(
        datetime(
            args.year,
            args.month,
            args.day,
            args.hour,
            args.minute,
            args.second,
        ),
        scale="utc",
    )

    # Equatorial coordinates for obs and SOLV
    obs_coord = SkyCoord(
        ra=args.obs_ra_deg * u.deg,
        dec=args.obs_dec_deg * u.deg,
        frame="cirs",
    )
    SOLV_coord = SkyCoord(
        ra=SOLV_ra_deg * u.deg,
        dec=SOLV_dec_deg * u.deg,
        frame="cirs",
    )

    altaz_frame = AltAz(obstime=obstime, location=location)
    obs_altaz = obs_coord.transform_to(altaz_frame)
    SOLV_altaz = SOLV_coord.transform_to(altaz_frame)

    obs_alt_deg = obs_altaz.alt.to(u.deg).value
    obs_az_deg = obs_altaz.az.to(u.deg).value
    SOLV_alt_deg = SOLV_altaz.alt.to(u.deg).value
    SOLV_az_deg = SOLV_altaz.az.to(u.deg).value

    alt_offset_deg = SOLV_alt_deg - obs_alt_deg
    az_offset_deg = SOLV_az_deg - obs_az_deg

    return alt_offset_deg, az_offset_deg, obs_alt_deg, obs_az_deg, SOLV_alt_deg, SOLV_az_deg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict RA/DEC acquisition offsets using trained XGBoost models."
    )

    # Time / date
    parser.add_argument("--year", type=int, required=True, help="UTC year")
    parser.add_argument("--month", type=int, required=True, help="UTC month (1-12)")
    parser.add_argument("--day", type=int, required=True, help="UTC day (1-31)")
    parser.add_argument("--hour", type=int, required=True, help="UTC hour (0-23)")
    parser.add_argument("--minute", type=int, required=True, help="UTC minute (0-59)")
    parser.add_argument("--second", type=int, required=True, help="UTC second (0-59)")

    # LST and obs coordinates (in degrees, LST in hours)
    parser.add_argument("--lst-hours", type=float, required=True, help="Local sidereal time in hours")
    parser.add_argument("--obs-ra-deg", type=float, required=True, help="Target obs RA in degrees")
    parser.add_argument("--obs-dec-deg", type=float, required=True, help="Target obs DEC in degrees")

    # Telescope location for Alt/Az conversion
    parser.add_argument(
        "--lat-deg",
        type=float,
        required=True,
        help="Telescope geodetic latitude in degrees (+N)",
    )
    parser.add_argument(
        "--lon-deg",
        type=float,
        required=True,
        help="Telescope geodetic longitude in degrees (+E)",
    )
    parser.add_argument(
        "--elevation-m",
        type=float,
        default=0.0,
        help="Telescope elevation in meters (default: 0.0)",
    )

    # Optional paths
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing model_ra.json and model_dec.json (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help=f"Path to prediction log file (default: {DEFAULT_LOG_FILE})",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load models (as before)
    model_ra = xgb.XGBRegressor()
    model_ra.load_model("models/model_ra.json")

    model_dec = xgb.XGBRegressor()
    model_dec.load_model("models/model_dec.json")

    # 2. Get previous acquisition information from log
    prev_vals = read_previous_from_log(args.log_file)

    # 3. Build feature vector
    X = build_feature_vector(args, prev_vals)

    # 4. Predict RA and DEC offsets
    ra_offset_pred = float(model_ra.predict(X)[0])
    dec_offset_pred = float(model_dec.predict(X)[0])

    # 5. Construct approximate SOLV coordinates for this acquisition (equatorial)
    SOLV_ra = args.obs_ra_deg - ra_offset_pred
    SOLV_dec = args.obs_dec_deg - dec_offset_pred

    # 6. Compute horizon-coordinate info and offsets
    (
        alt_offset_deg,
        az_offset_deg,
        obs_alt_deg,
        obs_az_deg,
        SOLV_alt_deg,
        SOLV_az_deg,
    ) = compute_horizon_offsets(args, SOLV_ra, SOLV_dec)

    # 7. Append to log for the next call
    append_to_log(
        args.log_file,
        SOLV_ra,
        SOLV_dec,
        ra_offset_pred,
        dec_offset_pred,
        args.obs_ra_deg,
        args.obs_dec_deg,
    )

    # 8. Print results
    print("=== Prediction ===")
    print(f"obs RA (deg):  {args.obs_ra_deg:.10f}")
    print(f"obs DEC (deg): {args.obs_dec_deg:.10f}")
    print()
    print(f"Predicted RA offset (deg):  {ra_offset_pred:.10f}")
    print(f"Predicted DEC offset (deg): {dec_offset_pred:.10f}")
    print()
    print(f"obs - Offset RA (deg): {SOLV_ra:.10f}")
    print(f"obs - Offset DEC (deg): {SOLV_dec:.10f}")
    print()
    print(f"obs ALT (deg): {obs_alt_deg:.10f}")
    print(f"obs AZ  (deg): {obs_az_deg:.10f}")
    print()
    print(f"Predicted AZ offset (deg):  {az_offset_deg:.10f}")
    print(f"Predicted ALT offset (deg): {alt_offset_deg:.10f}")
    print()
    print(f"obs - Offset AZ  (deg): {SOLV_az_deg:.10f}")
    print(f"obs - Offset ALT (deg): {SOLV_alt_deg:.10f}")
    print()
    print(f"Log file updated: {args.log_file}")


if __name__ == "__main__":
    main()


'''
Example

python predict.py \
  --year 2025 \
  --month 5 \
  --day 9 \
  --hour 8 \
  --minute 0 \
  --second 0 \
  --lst-hours 3.5 \
  --obs-ra-deg 123.456 \
  --obs-dec-deg -20.0 \
  --lat-deg 31.9583 \
  --lon-deg -111.5986 \
  --elevation-m 2400 \
  --models-dir models \
  --log-file prediction_log.txt
'''
