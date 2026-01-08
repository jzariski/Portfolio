#!/usr/bin/env python3
"""
prepare_and_train_xgb.py

1) Load and sort the HDF5‐saved array (year,month,day,hour,minute,second,
   lst_hours, tpt_ra_deg, tpt_dec_deg, wcs_ra_deg, wcs_dec_deg).
2) Split chronologically into train(70%), eval(10%), test(20%).
3) Build X (all columns except WCS RA/Dec) and y = [TPT−WCS].
4) Define and train two XGBRegressors (one for RA‐offset, one for DEC‐offset).
5) Predict on test, then plot & save true vs. predicted offsets.
"""

import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import h5py
import xgboost as xgb
import matplotlib.pyplot as plt


DATA_FILE = 'data/data.h5'

# -----------------------------------------------------------------------------
# 1) Load & sort utilities
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 2) Split into train/eval/test
# -----------------------------------------------------------------------------

def split_time_series(data, train_frac=0.7, eval_frac=0.1):
    """
    Chronologically split:
      - first train_frac → train
      - next eval_frac  → eval
      - remainder       → test
    """
    N = len(data)
    i_train = int(N * train_frac)
    i_eval  = i_train + int(N * eval_frac)
    return data[:i_train], data[i_train:i_eval], data[i_eval:]

# -----------------------------------------------------------------------------
# 3) Build X/y
# -----------------------------------------------------------------------------

## Reminder of what the feature matrix looks like
'''
 data = np.column_stack([
        cols['years'], cols['months'], cols['days'],
        cols['hours'], cols['minutes'], cols['seconds'],
        cols['lst'], cols['tpt_ra'], cols['tpt_dec'],
        cols['wcs_ra'], cols['wcs_dec']
    ])
'''


def make_features_and_labels(split_data):
    """
    Inputs X: all columns except the last two (WCS RA/Dec).
    Labels y: shape (M,2) = [TPT_RA - WCS_RA, TPT_DEC - WCS_DEC].
    Column indices:
       7:TPT_RA  8:TPT_DEC  9:WCS_RA  10:WCS_DEC
    """
    
    X = split_data[:, [0,1,2,3,4,5,6,7,8]]
    
    tpt_ra, tpt_dec = split_data[:,7], split_data[:,8]
    
    wcs_ra, wcs_dec = split_data[:,9], split_data[:,10]

    
    
    previous_acq_error_ra, previous_acq_error_dec = (tpt_ra - wcs_ra), (tpt_dec - wcs_dec)
    previous_acq_error_ra, previous_acq_error_dec = np.roll(previous_acq_error_ra, 1), np.roll(previous_acq_error_dec,1)
    previous_acq_error_ra[0] = 0
    previous_acq_error_dec[0] = 0
    
    

    X = np.column_stack([X, previous_acq_error_ra, previous_acq_error_dec])
    y = np.column_stack([tpt_ra - wcs_ra, tpt_dec - wcs_dec])
    
    
    '''
    ### What X looks like
    data = np.column_stack([
        years, months, days,
        hours, minutes, seconds,
        lst_hours,
        tpt_ra_deg, tpt_dec_deg,
        previous_acq_error_ra, previous_acq_error_dec,
    ])
    '''
    return X, y

# -----------------------------------------------------------------------------
# 4) XGBoost training & plotting (with save)
# -----------------------------------------------------------------------------

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
    
    model_ra.get_booster().save_model("models/model_ra.json")
    model_dec.get_booster().save_model("models/model_dec.json")
    
    
    # Predict on test set
    ra_pred  = model_ra.predict(X_test)
    dec_pred = model_dec.predict(X_test)
    
    # --- compute absolute errors ---
    err_ra  = np.abs(ra_pred  - y_test[:, 0])
    err_dec = np.abs(dec_pred - y_test[:, 1])
    
    # --- compute true‐value magnitudes ---
    mag_ra  = np.abs(y_test[:, 0])
    mag_dec = np.abs(y_test[:, 1])
    

    # --- print RA stats ---
    print("=== RA Offset Performance ===")
    print(f"Median absolute offset prediction error RA (Arcsec)     : {np.median(err_ra)*3600:.4f} arcsec")
    print(f"90th-percentile offset prediction error RA (Arcsec) : {np.percentile(err_ra, 90)*3600:.4f} arcsec")
    print(f"Median magnitude of TPT-WCS RA (Arcsec)      : {np.median(mag_ra)*3600:.4f} arcsec")
    print(f"90th-percentile of TPT-WCS RA (Arcsec).  : {np.percentile(mag_ra, 90)*3600:.4f} arcsec\n")
    
    # --- print DEC stats ---
    print("=== DEC Offset Performance ===")
    print(f"Median absolute offset prediction error Dec (Arcsec)      : {np.median(err_dec)*3600:.4f} arcsec")
    print(f"90th-percentile offset prediction error Dec (Arcsec) : {np.percentile(err_dec, 90)*3600:.4f} arcsec")
    print(f"Median magnitude of TPT-WCS Dec (Arcsec)      : {np.median(mag_dec)*3600:.4f} arcsec")
    print(f"90th-percentile of TPT-WCS Dec (Arcsec)  : {np.percentile(mag_dec, 90)*3600:.4f} arcsec")

    # --- Plot and save RA offsets ---
    plt.figure(figsize=(15,10))
    plt.suptitle('Offsets on Test Set')
    plt.subplot(1,2,1)
    
    
    
    ## CDF Plotting
    plt.figure(figsize=(15,10))
    plt.suptitle('CDF of Error in Predicting Offset')
    plt.subplot(1,2,1)
    eps = 1e-10
    plt.hist(np.log10(abs(mag_ra) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Offset RA')
    plt.hist(np.log10(abs(err_ra) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Error in Predicting Offset RA')
    plt.title('CDF of Log Absolute Error in RA Offset')
    plt.xlabel('Log Absolute Error (Degrees)')
    plt.ylabel('Percentage of Points')
    plt.xlim([-5,-1.0])
    plt.legend()
    
    plt.subplot(1,2,2)
    eps = 1e-10
    plt.hist(np.log10(abs(mag_dec) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Offset Dec')
    plt.hist(np.log10(abs(err_dec) + eps), bins=1000, density=True, cumulative=True, histtype='step', label='Error in Predicting Offset Dec')
    plt.title('CDF of Log Absolute Error in Dec Offset')
    plt.xlabel('Log Absolute Error (Degrees)')
    plt.ylabel('Percentage of Points')
    plt.xlim([-5,-1.5])
    plt.legend()

    plt.savefig('plots/cdf.png', dpi=150)  # save to PNG


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    with h5py.File(DATA_FILE, 'r') as hf:
        data =  hf['data'][:]
        
    data = sort_by_datetime(data)
    
    X, y = make_features_and_labels(data)

    # Split
    train_data, eval_data, test_data = split_time_series(data)

    # Build X/y
    X_train, y_train = make_features_and_labels(train_data)
    X_eval,  y_eval  = make_features_and_labels(eval_data)
    X_test,  y_test  = make_features_and_labels(test_data)

    # Show shapes
    print(f"Train:  X={X_train.shape}, y={y_train.shape}")
    print(f"Eval:   X={X_eval.shape},  y={y_eval.shape}")
    print(f"Test:   X={X_test.shape},  y={y_test.shape}\n")

    # Train models, plot, and save figures
    train_and_evaluate(X_train, y_train, X_eval, y_eval, X_test, y_test)

if __name__ == "__main__":
    main()