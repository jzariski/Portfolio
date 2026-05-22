#!/usr/bin/env python3
"""
Sobol.py

Run Sobol sensitivity analysis against the trained RA/Dec XGBoost models.

This script intentionally uses the same 11-feature order as booster.py and
predict.py:
  year, month, day, hour, minute, second, lst_hours,
  obs_ra_deg, obs_dec_deg, previous_acq_error_ra, previous_acq_error_dec
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import xgboost as xgb
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample


DATA_FILE = "data/data.h5"
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

FEATURE_NAMES = (
    "year", "month", "day",
    "hour", "minute", "second",
    "lst_hours",
    "obs_ra_deg", "obs_dec_deg",
    "previous_acq_error_ra", "previous_acq_error_dec",
)


def load_data(h5_file_path):
    """Load the numeric training array saved by load_out.py."""
    with h5py.File(h5_file_path, "r") as hf:
        return hf["data"][:]


def sort_by_datetime(data):
    """Sort rows by UTC timestamp columns before creating lagged features."""
    years, months, days = data[:, 0], data[:, 1], data[:, 2]
    hours, mins, secs = data[:, 3], data[:, 4], data[:, 5]
    idx = np.lexsort((secs, mins, hours, days, months, years))
    return data[idx]


def split_time_series(data, train_frac=0.7, eval_frac=0.1):
    """Return chronological train/eval/test row blocks."""
    n_rows = len(data)
    i_train = int(n_rows * train_frac)
    i_eval = i_train + int(n_rows * eval_frac)
    return data[:i_train], data[i_train:i_eval], data[i_eval:]


def make_features_and_labels(data):
    """
    Match booster.py exactly so Sobol results describe the deployed model.

    The last two features are autoregressive previous-error terms. They are
    created by shifting the current obs-solv error back by one row.
    """
    base_features = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    obs_ra, obs_dec = data[:, 7], data[:, 8]
    solv_ra, solv_dec = data[:, 9], data[:, 10]
    error_ra = obs_ra - solv_ra
    error_dec = obs_dec - solv_dec

    previous_error_ra = np.roll(error_ra, 1)
    previous_error_dec = np.roll(error_dec, 1)
    previous_error_ra[0] = 0.0
    previous_error_dec[0] = 0.0

    X = np.column_stack([base_features, previous_error_ra, previous_error_dec])
    y = np.column_stack([error_ra, error_dec])
    return X, y


def build_problem(X_train):
    """Use train-set min/max bounds for each model input feature."""
    bounds = []
    for i, name in enumerate(FEATURE_NAMES):
        lo = float(X_train[:, i].min())
        hi = float(X_train[:, i].max())

        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite bounds for {name}: lo={lo}, hi={hi}")

        if lo == hi:
            print(f"Warning: {name} is constant at {lo}; expanding bounds slightly.")
            eps = 1e-6 if lo == 0 else 1e-6 * abs(lo)
            lo -= eps
            hi += eps

        bounds.append([lo, hi])

    return {
        "num_vars": len(FEATURE_NAMES),
        "names": list(FEATURE_NAMES),
        "bounds": bounds,
    }


def print_indices(label, indices):
    """Pretty-print first-order and total-order sensitivity indices."""
    print(f"{label} Sobol indices")
    for name, s1, st, c1, ct in zip(
        FEATURE_NAMES,
        indices["S1"],
        indices["ST"],
        indices["S1_conf"],
        indices["ST_conf"],
    ):
        print(f"{name:>24s} | S1 = {s1:.4f} +/- {c1:.4f} | ST = {st:.4f} +/- {ct:.4f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sobol sensitivity analysis for trained models.")
    parser.add_argument("--data-file", default=DATA_FILE, help=f"HDF5 training data (default: {DATA_FILE})")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help=f"Model directory (default: {MODELS_DIR})")
    parser.add_argument("--output", default=str(LOGS_DIR / "sobol_output.txt"), help="Text output path")
    parser.add_argument("--n-base", type=int, default=2**14, help="Saltelli base sample size")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(output_path, "w")

    data = sort_by_datetime(load_data(args.data_file))
    X, y = make_features_and_labels(data)
    train_data, _, _ = split_time_series(data)
    X_train, _ = make_features_and_labels(train_data)

    print(f"Data shape: {data.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Mean absolute RA offset in dataset: {np.mean(np.abs(y[:, 0])):.8f} deg")
    print(f"Mean absolute Dec offset in dataset: {np.mean(np.abs(y[:, 1])):.8f} deg")
    print()

    models_dir = Path(args.models_dir)
    model_ra = xgb.XGBRegressor()
    model_ra.load_model(str(models_dir / "model_ra.json"))
    model_dec = xgb.XGBRegressor()
    model_dec.load_model(str(models_dir / "model_dec.json"))

    problem = build_problem(X_train)
    param_values = sobol_sample.sample(problem, args.n_base, calc_second_order=False)

    Y_ra = model_ra.predict(param_values)
    Y_dec = model_dec.predict(param_values)

    print_indices(
        "RA",
        sobol_analyze.analyze(problem, Y_ra, calc_second_order=False, print_to_console=False),
    )
    print_indices(
        "Dec",
        sobol_analyze.analyze(problem, Y_dec, calc_second_order=False, print_to_console=False),
    )


if __name__ == "__main__":
    main()
