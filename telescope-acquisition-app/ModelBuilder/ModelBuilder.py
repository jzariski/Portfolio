import argparse
import sys
import numpy as np
import h5py
from datetime import datetime, timezone
import xgboost as xgb
import matplotlib.pyplot as plt

class ModelBuilder:
    
    '''
    Creates the XGBoost Model with desired parameters
    STILL NEED TO DECIDE HOW TO ACCEPT PARAMS
    For now params will just be [estimators, max_depth]

    Returns tuple of (XGBoost Model, lat lon)?

    '''

    def __init__(self, params, data, title, extraAugments=True):
        """
        Constructor / initializer.
        :params: Dictionary of parameters for the XGBoost model
        """
        # Public instance variable
        self.params = params
        self.data = data
        self.title = title # should be a string

        self.X_train = None
        self.Y_train = None

        self.X_eval = None
        self.Y_eval = None
        
        self.X_test = None
        self.Y_test = None

        self.extraAugments = extraAugments

        self.model = None

        self.preds = None

    # Note input should be sorted in advance!!!
    def _augmentData(self, sortedData):

        currData = np.copy(sortedData)
        mountAz, mountAlt = currData[:,6], currData[:,7]
        actualAz, actualAlt = currData[:,8], currData[:,9]
    
    
        previous_acq_error_az, previous_acq_error_alt = (mountAz - actualAz), (mountAlt - actualAlt)
        previous_acq_error_az, previous_acq_error_alt = np.roll(previous_acq_error_az, 1), np.roll(previous_acq_error_alt,1)
        previous_acq_error_az[0] = 0
        previous_acq_error_alt[0] = 0
        
        previous_acq_az, previous_acq_alt = np.roll(mountAz, 1), np.roll(mountAlt,1)
        previous_acq_az[0] = 0
        previous_acq_alt[0] = 0

        # Appends four new features to the end of the sorted total data array
        self.data = np.column_stack([self.data, previous_acq_error_az, previous_acq_error_alt, previous_acq_az, previous_acq_alt])
    
    '''
    Split data into training for the model
    '''
    def TrainTestSplit(self, split_by_Time=True, train_pct=0.7, eval_pct=0.8):
        # Do a split by date
        if split_by_Time:
            sort_idx = np.lexsort((
                self.data[:, 5],  # second
                self.data[:, 4],  # minute
                self.data[:, 3],  # hour
                self.data[:, 2],  # day
                self.data[:, 1],  # month
                self.data[:, 0],  # year
            ))
            # apply it to get the sorted array
            self.data = self.data[sort_idx]
        
        # Augment the data with feature engineering
        if self.extraAugments:
            self._augmentData(self.data)
        
        # Offset between Mount Az/El and Actual Az/El
        # Index 6/7 is Mount Az/El and index 8/9 is Actual Az/El
        X_idx = [i for i in range(self.data.shape[1]) if i not in {8, 9}]
        X = self.data[:,X_idx]
        Y = self.data[:,[6,7]] - self.data[:,[8,9]]

        train_idx = int(np.round(self.data.shape[0] * train_pct))
        eval_idx = int(np.round(self.data.shape[0] * eval_pct))

        if train_idx == eval_idx:
            raise Exception('Train/Test Split Index equal, data set probably too small or percentages are equal.')
        
        X_train, Y_train = X[0:train_idx,:], Y[0:train_idx,:]
        X_eval, Y_eval = X[train_idx:eval_idx,:], Y[train_idx:eval_idx,:]
        X_test, Y_test = X[eval_idx:,:], Y[eval_idx:,:]
        print(X_test.shape)
        
        self.X_train, self.Y_train, self.X_eval, self.Y_eval, self.X_test, self.Y_test = X_train, Y_train, X_eval, Y_eval, X_test, Y_test


    '''
    Creates the XGBoost Model and trains it with desired X_train, Y_Train
    '''
    def createModel(self):

        xgb_params = {
            'n_estimators': self.params['n_estimators'],
            'max_depth': self.params['max_depth'],
            'learning_rate': self.params['learning_rate'],
            'objective': 'reg:absoluteerror',
            'verbosity': 1,
            'early_stopping_rounds': self.params['early_stopping_rounds']
        }


        self.model = xgb.XGBRegressor(**xgb_params)
        self.model.fit(
            self.X_train, self.Y_train,
            eval_set=[(self.X_eval, self.Y_eval)],
            verbose=True,
        )

    def evaluateModel(self):
        self.preds = self.model.predict(self.X_test)
    
    def make_cdf_figure(self, offsetAz, errorAz, offsetAlt, errorAlt, eps=1e-8):
        """
        Returns a matplotlib Figure containing two CDF plots
        of log‐absolute errors in azimuth and altitude offsets.
        """
        # Create a figure + 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

        # Left: Azimuth CDF
        ax1.hist(
            np.log10(np.abs(offsetAz) + eps),
            bins=1000,
            density=True,
            cumulative=True,
            histtype="step",
            label="Offset Azimuth"
        )
        ax1.hist(
            np.log10(np.abs(errorAz) + eps),
            bins=1000,
            density=True,
            cumulative=True,
            histtype="step",
            label="Error in Predicting Offset Azimuth"
        )
        ax1.set_title("CDF of Log Absolute Error in Azimuth Offset")
        ax1.set_xlabel("Log Absolute Error (Degrees)")
        ax1.set_ylabel("Percentage of Points")
        ax1.legend()

        # Right: Altitude CDF
        ax2.hist(
            np.log10(np.abs(offsetAlt) + eps),
            bins=1000,
            density=True,
            cumulative=True,
            histtype="step",
            label="Offset Altitude"
        )
        ax2.hist(
            np.log10(np.abs(errorAlt) + eps),
            bins=1000,
            density=True,
            cumulative=True,
            histtype="step",
            label="Error in Predicting Offset Altitude"
        )
        ax2.set_title("CDF of Log Absolute Error in Altitude Offset")
        ax2.set_xlabel("Log Absolute Error (Degrees)")
        ax2.set_ylabel("Percentage of Points")
        ax2.legend()

        # Tidy up layout
        fig.tight_layout()
        return fig
    
    def createCDF(self):
        offsetMagnitude = np.abs(self.Y_test)
        error = self.preds - self.Y_test
        offsetAz, offsetAlt = offsetMagnitude[:,0], offsetMagnitude[:,1]
        errorAz, errorAlt = error[:,0], error[:,1]
        eps = 1e-10

        return self.make_cdf_figure(offsetAz, errorAz, offsetAlt, errorAlt, eps)
    
    def displayStatistics(self):
        offsetMagnitude = np.abs(self.Y_test)
        error = self.preds - self.Y_test
        offsetAz, offsetAlt = offsetMagnitude[:,0], offsetMagnitude[:,1]
        errorAz, errorAlt = error[:,0], error[:,1]

        return offsetAz, offsetAlt, errorAz, errorAlt
    
    '''
    Features needed depend on input file
    STILL NEED TO MAKE THIS WORK
    '''

    def formatPrediction(self, mountAz, mountAlt, 
                         second, minute, hour, day, month, year,
                         currList, extraOn):
        inputPred = [year, month, day, hour, minute, second, mountAz, mountAlt]
        if extraOn:
            if len(currList) > 0:
                previous_acq_error_az, previous_acq_error_alt, previous_acq_az, previous_acq_alt = currList[-1]
            else:
                previous_acq_error_az, previous_acq_error_alt, previous_acq_az, previous_acq_alt = 0, 0, 0, 0
            inputPred.append(previous_acq_error_az)
            inputPred.append(previous_acq_error_alt)
            inputPred.append(previous_acq_az)
            inputPred.append(previous_acq_alt)
        return np.asarray([inputPred])

    def updatePastList(self, currList, inputAz, inputEl, actualAz, actualAlt):
        previous_acq_error_az, previous_acq_error_alt = inputAz - actualAz, inputAlt - actualAlt
        previous_acq_az, previous_acq_alt = inputAz, inputAlt
        currList.append((previous_acq_error_az, previous_acq_error_alt, previous_acq_az, previous_acq_alt))


    def makePrediction(self, mountAz, mountAlt, 
                         second, minute, hour, day, month, year,
                         currList, extraOn):
        if self.model:

            input_pred = self.formatPrediction(mountAz, mountAlt,
                                second, minute, hour, day, month, year,
                                currList, extraOn)
            
            outs = self.model.predict(input_pred)
            return outs
        else:
            print("No Model Trained")
            return None

