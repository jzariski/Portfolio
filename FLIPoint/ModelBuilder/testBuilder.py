import argparse
import sys
import numpy as np
import h5py
from datetime import datetime, timezone
import xgboost as xgb
import matplotlib.pyplot as plt

import ModelBuilder

input_file = '../Parsing/lco_output.h5'

with h5py.File(input_file, 'r') as h5f:
    # Load observations array
    data = h5f['observations'][:]
    # Load metadata
    lon_attr = h5f.attrs.get('longitude_deg', None)
    lat_attr = h5f.attrs.get('latitude_deg', None)
    lon_ds = h5f['longitude_deg'][()]
    lat_ds = h5f['latitude_deg'][()]
    # Load column headers (string dataset)
    col_headers = h5f['column_headers'][:]
    # Decode bytes to str if necessary
    if isinstance(col_headers[0], bytes):
        col_headers = [h.decode('utf-8') for h in col_headers]
    else:
        col_headers = list(col_headers)


params = {
    'n_estimators': 100,
    'max_depth': 3
}

worker = ModelBuilder.ModelBuilder(params, data, 'LCO', False)

print('Total Data shape', worker.data.shape)
worker.TrainTestSplit()
print('Shapes: Train, Eval, Test')
print(worker.X_train.shape, worker.Y_train.shape)
print(worker.X_eval.shape, worker.Y_eval.shape)
print(worker.X_test.shape, worker.Y_test.shape)
print('Testing if Training works')
worker.createModel()
print('Testing if plot creator works')
worker.evaluateModel()


