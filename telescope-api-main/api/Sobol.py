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

from SALib.sample import saltelli
from SALib.analyze import sobol

import numpy as np
import h5py
import xgboost as xgb
import matplotlib.pyplot as plt
sys.stdout = open("logs/sobol_output.txt", "w")
# -----------------------------------------------------------------------------
# 1) Load & sort utilities
# -----------------------------------------------------------------------------

DATA_FILE = 'data/data.h5'


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
    
    ## USING WCS AS A FEAUTRE BECAUSE WE'RE TRYING TO HIT IT
    X = split_data[:, [0,1,2,3,4,5,6,9,10]]
    
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
# Entry point
# -----------------------------------------------------------------------------


# Load & sort
data = load_data(DATA_FILE)
data = sort_by_datetime(data)

print('Data shape')
print(data.shape)

X, y = make_features_and_labels(data)

# Split
train_data, eval_data, test_data = split_time_series(data)

# Build X/y
X_train, y_train = make_features_and_labels(train_data)
X_eval,  y_eval  = make_features_and_labels(eval_data)
X_test,  y_test  = make_features_and_labels(test_data)

print(np.mean(abs(y_test[:,0])))
print(np.mean(abs(y_test[:,1])))


# Show shapes
print(f"Train:  X={X_train.shape}, y={y_train.shape}")
print(f"Eval:   X={X_eval.shape},  y={y_eval.shape}")
print(f"Test:   X={X_test.shape},  y={y_test.shape}\n")


model_ra = xgb.XGBRegressor()
model_ra.load_model('models/model_ra.json')

model_dec = xgb.XGBRegressor()
model_dec.load_model('models/model_dec.json')


# -----------------------------------------------------------------------------
# 1. Prepare your data & model
# -----------------------------------------------------------------------------

# Assume you have:
#   X_train: (N_train, D) array of training inputs
#   model: a trained model with a .predict(X) method
# If your model uses a different interface (e.g. predict_proba or a torch model),
# adjust the call below.

# Example placeholders – replace these with your actual data/model:
# X_train = ...
# model = ...

# Number of features
D = X_train.shape[1]

# -----------------------------------------------------------------------------
# 2. Define the SALib problem
# -----------------------------------------------------------------------------

# Feature names (optional, for readability in outputs)
names = ['years', 'months', 'days',
        'hours', 'minutes', 'seconds',
        'lst_hours',
        'tpt_ra_deg', 'tpt_dec_deg',
        'previous_acq_error_ra', 'previous_acq_error_dec']

print('here?')
bounds = []
for i in range(D):
    lo = float(X_train[:, i].min())
    hi = float(X_train[:, i].max())

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Non-finite bounds for {names[i]}: lo={lo}, hi={hi}")

    if lo == hi:
        # Warn so you know which ones are degenerate
        print(f"Warning: {names[i]} has constant value {lo} in X_train; expanding bounds.")
        eps = 1e-6 if lo == 0 else 1e-6 * abs(lo)
        lo -= eps
        hi += eps

    if lo >= hi:
        raise ValueError(f"Still bad bounds for {names[i]}: lo={lo}, hi={hi}")

    bounds.append([lo, hi])
        
        
# Bounds for each feature. Here we use the min/max from the training set;
# you could instead specify domain knowledge bounds.
#bounds = [[X_train[:, i].min(), X_train[:, i].max()] for i in range(D)]

problem = {
    'num_vars': D,
    'names':    names,
    'bounds':   bounds
}

# -----------------------------------------------------------------------------
# 3. Generate Saltelli samples
# -----------------------------------------------------------------------------

# Base sample size (choose based on your compute budget; 1000–5000 is typical)
N_BASE = 2**14

# NOTE: setting calc_second_order=True will compute S_ij as well, but is more expensive
param_values = saltelli.sample(problem, N_BASE, calc_second_order=False)

# -----------------------------------------------------------------------------
# 4. Evaluate the model on the sample matrix
# -----------------------------------------------------------------------------

# If your model.predict accepts batches, great; otherwise you can loop.
# This will produce an array y of length len(param_values)
Y_ra = model_ra.predict(param_values)
Y_dec = model_dec.predict(param_values)

# -----------------------------------------------------------------------------
# 5. Compute Sobol' indices
# -----------------------------------------------------------------------------

print('RA Sobol indices')
Si_ra = sobol.analyze(
    problem,
    Y_ra,
    calc_second_order=False,  # True if you want 2nd-order indices S_ij
    print_to_console=True      # set False if you prefer to process Si yourself
)

print('Dec Sobol indices')
Si_dec = sobol.analyze(
    problem,
    Y_dec,
    calc_second_order=False,  # True if you want 2nd-order indices S_ij
    print_to_console=True      # set False if you prefer to process Si yourself
)

# -----------------------------------------------------------------------------
# 6. Inspect & use the results
# -----------------------------------------------------------------------------

# First‐order (main effect) indices
S1_ra = Si_ra['S1']      # array of length D
# Total‐order indices
ST_ra = Si_ra['ST']      # array of length D

# First‐order (main effect) indices
S1_dec = Si_dec['S1']      # array of length D
# Total‐order indices
ST_dec = Si_dec['ST']      # array of length D


# (Optional) Confidence intervals
S1_conf_ra = Si_ra['S1_conf']
ST_conf_ra = Si_ra['ST_conf']

S1_conf_dec = Si_dec['S1_conf']
ST_conf_dec = Si_dec['ST_conf']

print('Ra Sobol Indices')

# Pretty‐print
for name, s1, st, c1, ct in zip(names, S1_ra, ST_ra, S1_conf_ra, ST_conf_ra):
    print(f"{name:>3s} | S1 = {s1:.4f} ± {c1:.4f}  | ST = {st:.4f} ± {ct:.4f}")

print()
print('Dec Sobol Indices')
print()

for name, s1, st, c1, ct in zip(names, S1_dec, ST_dec, S1_conf_dec, ST_conf_dec):
    print(f"{name:>3s} | S1 = {s1:.4f} ± {c1:.4f}  | ST = {st:.4f} ± {ct:.4f}")

