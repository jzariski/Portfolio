#!/usr/bin/env python3
"""
booster.py

1) Load and sort the HDF5-saved array (year,month,day,hour,minute,second,
   lst_hours, obs_ra_deg, obs_dec_deg, solv_ra_deg, solv_dec_deg).
2) Split chronologically into train(70%), eval(10%), test(20%).
3) Build autoregressive X and y = [obs - solv].
4) Define and train two XGBRegressors (one for RA-offset, one for DEC-offset).
5) Predict on test, then plot & save true vs. predicted offsets.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import h5py
import xgboost as xgb

# Keep Matplotlib cache writes inside the project/sandbox when the user home
# directory is not writable.
Path("plots").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("plots") / ".matplotlib"))
import matplotlib.pyplot as plt


DATA_FILE = "data/data.h5"
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

FEATURE_NAMES = (
    "year", "month", "day",
    "hour", "minute", "second",
    "lst_hours",
    "obs_ra_deg", "obs_dec_deg",
    "previous_acq_error_ra", "previous_acq_error_dec",
)

# 1) Load & sort utilities

def load_data(h5_file_path):
    """Load array from HDF5 file under dataset 'data'."""
    with h5py.File(h5_file_path, 'r') as hf:
        return hf['data'][:]

def sort_by_datetime(data):
    """
    Lexicographically sort rows by the first six columns
    [year, month, day, hour, minute, second].
    """
    years, months, days = data[:,0], data[:,1], data[:,2]
    hours, mins, secs   = data[:,3], data[:,4], data[:,5]
    idx = np.lexsort((secs, mins, hours, days, months, years))
    return data[idx]

def make_features_and_labels(split_data):
    """
    Build model inputs and labels in the same order used by predict.py.

    Inputs X:
      year, month, day, hour, minute, second,
      lst_hours, obs_ra_deg, obs_dec_deg,
      previous_acq_error_ra, previous_acq_error_dec

    Labels y: shape (M,2) = [obs_RA - solv_RA, obs_DEC - solv_DEC].

    The final two input columns make the model autoregressive: every row gets
    the previous acquisition's true pointing error. At inference time
    predict.py uses the previous predicted error from the log as the available
    proxy for that same information.

    Column indices:
       7:obs_RA  8:obs_DEC  9:solv_RA  10:solv_DEC
    """
    base_features = split_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    obs_ra, obs_dec = split_data[:, 7], split_data[:, 8]
    solv_ra, solv_dec = split_data[:, 9], split_data[:, 10]

    current_error_ra = obs_ra - solv_ra
    current_error_dec = obs_dec - solv_dec

    previous_error_ra = np.roll(current_error_ra, 1)
    previous_error_dec = np.roll(current_error_dec, 1)
    previous_error_ra[0] = 0.0
    previous_error_dec[0] = 0.0

    X = np.column_stack([base_features, previous_error_ra, previous_error_dec])
    y = np.column_stack([obs_ra - solv_ra, obs_dec - solv_dec])
    return X, y


def split_features_and_labels(X, y, train_frac=0.7, eval_frac=0.1):
    """
    Split already-built autoregressive features chronologically.

    Building X before this split preserves the true previous acquisition at
    the train/eval/test boundaries. Resetting previous_error to zero at each
    split would weaken the autoregressive behavior exactly where evaluation
    begins.
    """
    n_rows = len(X)
    i_train = int(n_rows * train_frac)
    i_eval = i_train + int(n_rows * eval_frac)
    return (
        X[:i_train], y[:i_train],
        X[i_train:i_eval], y[i_train:i_eval],
        X[i_eval:], y[i_eval:],
    )

# 4) XGBoost training & plotting (with save)

# Tweakable XGBoost parameters:
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 3,
    'objective': 'reg:absoluteerror',
    'verbosity': 0,
    'early_stopping_rounds': 100
}

def train_and_evaluate(X_train, y_train, X_eval, y_eval, X_test, y_test):
    """
    Train two XGBRegressors (RA & DEC), predict on test,
    plot true vs predicted offsets, and save the figures.
    """
    # Train RA offset model
    model_ra = xgb.XGBRegressor(**xgb_params)
    model_ra.fit(
        X_train, y_train[:,0],
        eval_set=[(X_eval, y_eval[:,0])],
        verbose=False,
    )

    # Train DEC offset model
    model_dec = xgb.XGBRegressor(**xgb_params)
    model_dec.fit(
        X_train, y_train[:,1],
        eval_set=[(X_eval, y_eval[:,1])],
        verbose=False
    )
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    model_ra.save_model(str(MODELS_DIR / "model_ra.json"))
    model_dec.save_model(str(MODELS_DIR / "model_dec.json"))
    
    
    # Predict on test set
    ra_pred  = model_ra.predict(X_test)
    dec_pred = model_dec.predict(X_test)
    
    # --- compute absolute errors ---
    err_ra  = np.abs(ra_pred  - y_test[:, 0])
    err_dec = np.abs(dec_pred - y_test[:, 1])
    
    # --- compute true-value magnitudes ---
    mag_ra  = np.abs(y_test[:, 0])
    mag_dec = np.abs(y_test[:, 1])
    

    # Print RA stats.
    print("=== RA Offset Performance ===")
    print(f"Median absolute offset prediction error RA (Arcsec)     : {np.median(err_ra)*3600:.4f} arcsec")
    print(f"90th-percentile offset prediction error RA (Arcsec) : {np.percentile(err_ra, 90)*3600:.4f} arcsec")
    print(f"Median magnitude of obs-solv RA (Arcsec)      : {np.median(mag_ra)*3600:.4f} arcsec")
    print(f"90th-percentile of obs-solv RA (Arcsec).  : {np.percentile(mag_ra, 90)*3600:.4f} arcsec\n")
    
    # Print Dec stats.
    print("=== DEC Offset Performance ===")
    print(f"Median absolute offset prediction error Dec (Arcsec)      : {np.median(err_dec)*3600:.4f} arcsec")
    print(f"90th-percentile offset prediction error Dec (Arcsec) : {np.percentile(err_dec, 90)*3600:.4f} arcsec")
    print(f"Median magnitude of obs-solv Dec (Arcsec)      : {np.median(mag_dec)*3600:.4f} arcsec")
    print(f"90th-percentile of obs-solv Dec (Arcsec)  : {np.percentile(mag_dec, 90)*3600:.4f} arcsec")

    # Plot and save RA offsets ---
    # CDFs show whether the model improves over simply applying no correction.
    plt.figure(figsize=(15,10))
    plt.suptitle('CDF of Error in Predicting Offset')
    plt.subplot(1,2,1)
    eps = 1e-10
    plt.hist(np.log10(np.abs(mag_ra) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Offset RA')
    plt.hist(np.log10(np.abs(err_ra) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Error in Predicting Offset RA')
    plt.title('CDF of Log Absolute Error in RA Offset')
    plt.xlabel('Log Absolute Error (Degrees)')
    plt.ylabel('Percentage of Points')
    plt.xlim([-5,-1.0])
    plt.legend()
    
    plt.subplot(1,2,2)
    eps = 1e-10
    plt.hist(np.log10(np.abs(mag_dec) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Offset Dec')
    plt.hist(np.log10(np.abs(err_dec) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Error in Predicting Offset Dec')
    plt.title('CDF of Log Absolute Error in Dec Offset')
    plt.xlabel('Log Absolute Error (Degrees)')
    plt.ylabel('Percentage of Points')
    plt.xlim([-5,-1.5])
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cdf.png", dpi=150)


def parse_args():
    parser = argparse.ArgumentParser(description="Train autoregressive XGBoost pointing-offset models.")
    parser.add_argument("--data-file", default=DATA_FILE, help=f"HDF5 file from load_out.py (default: {DATA_FILE})")
    return parser.parse_args()


# Entry point

## Test

def main():
    args = parse_args()
    data = load_data(args.data_file)
    data = sort_by_datetime(data)

    # Build autoregressive features once over the full chronological sequence.
    # This preserves previous acquisition context across split boundaries.
    X, y = make_features_and_labels(data)
    X_train, y_train, X_eval, y_eval, X_test, y_test = split_features_and_labels(X, y)

    # Show shapes
    print(f"Train:  X={X_train.shape}, y={y_train.shape}")
    print(f"Eval:   X={X_eval.shape},  y={y_eval.shape}")
    print(f"Test:   X={X_test.shape},  y={y_test.shape}\n")

    # Train models, plot, and save figures
    train_and_evaluate(X_train, y_train, X_eval, y_eval, X_test, y_test)

if __name__ == "__main__":
    main()
