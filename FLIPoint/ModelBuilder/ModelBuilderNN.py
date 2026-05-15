import argparse
import sys
import warnings
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.exceptions import ConvergenceWarning

class ModelBuilderNN:
    
    '''
    Creates the Neural Network Model with desired parameters
    STILL NEED TO DECIDE HOW TO ACCEPT PARAMS
    For now params will just be [estimators, max_depth]

    Returns tuple of (XGBoost Model, lat lon)?

    '''

    def __init__(self, params, data, title, extraAugments=True, headers=None):
        """
        Constructor / initializer.
        :params: Dictionary of parameters for the XGBoost model
        """
        # Public instance variable
        self.params = params
        self.data = data
        self.title = title # should be a string
        self.headers = headers or []

        self.X_train = None
        self.Y_train = None

        self.X_eval = None
        self.Y_eval = None
        
        self.X_test = None
        self.Y_test = None

        self.extraAugments = extraAugments

        self.model = None

        self.preds = None
        self.x_idx = None
        self.feature_names = []
        self.custom_feature_names = []
        self.feature_defaults = {}
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    # Note input should be sorted in advance!!!
    def _augmentData(self, sortedData):

        currData = np.copy(sortedData)
        mountRA, mountDec = currData[:,6], currData[:,7]
        actualRA, actualDec = currData[:,8], currData[:,9]
    
    
        previous_acq_error_RA, previous_acq_error_Dec = (mountRA - actualRA), (mountDec - actualDec)
        previous_acq_error_RA, previous_acq_error_Dec = np.roll(previous_acq_error_RA, 1), np.roll(previous_acq_error_Dec,1)
        previous_acq_error_RA[0] = 0
        previous_acq_error_Dec[0] = 0
        
        previous_acq_RA, previous_acq_Dec = np.roll(mountRA, 1), np.roll(mountDec,1)
        previous_acq_RA[0] = 0
        previous_acq_Dec[0] = 0

        # Appends four autoregressive features to the end of the sorted data array.
        self.data = np.column_stack([currData, previous_acq_error_RA, previous_acq_error_Dec, previous_acq_RA, previous_acq_Dec])
    
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
        
        if not 0 < train_pct < eval_pct < 1:
            raise ValueError("Splits must satisfy 0 < train < validation < 100%.")

        # Offset between Mount RA/El and Actual RA/El
        # Index 6/7 is Mount RA/El and index 8/9 is Actual RA/El
        X_idx = [i for i in range(self.data.shape[1]) if i not in {8, 9}]
        X = self.data[:,X_idx]
        Y = self.data[:,[6,7]] - self.data[:,[8,9]]

        train_idx = int(np.floor(self.data.shape[0] * train_pct))
        eval_idx = int(np.floor(self.data.shape[0] * eval_pct))

        if train_idx <= 0 or eval_idx <= train_idx or eval_idx >= self.data.shape[0]:
            raise ValueError("Split leaves an empty train, validation, or test set. Adjust percentages or use more data.")
        
        X_train, Y_train = X[0:train_idx,:], Y[0:train_idx,:]
        X_eval, Y_eval = X[train_idx:eval_idx,:], Y[train_idx:eval_idx,:]
        X_test, Y_test = X[eval_idx:,:], Y[eval_idx:,:]
        print(X_test.shape)
        
        base_headers = self.headers if len(self.headers) == (self.data.shape[1] - (4 if self.extraAugments else 0)) else [
            f"feature_{i}" for i in range(self.data.shape[1] - (4 if self.extraAugments else 0))
        ]
        all_headers = list(base_headers)
        if self.extraAugments:
            all_headers.extend([
                "previous_acq_error_RA",
                "previous_acq_error_Dec",
                "previous_mount_RA",
                "previous_mount_Dec",
            ])

        self.x_idx = X_idx
        self.feature_names = [all_headers[i] for i in X_idx]
        self.custom_feature_names = list(all_headers[10:len(base_headers)])
        self.feature_defaults = {
            all_headers[i]: float(np.nanmedian(self.data[:, i]))
            for i in range(10, len(base_headers))
        }
        self.X_train, self.Y_train, self.X_eval, self.Y_eval, self.X_test, self.Y_test = X_train, Y_train, X_eval, Y_eval, X_test, Y_test


    '''
    Creates the XGBoost Model and trains it with desired X_train, Y_Train
    '''
    def createModel(self, progress_callback=None):
        total_epochs = max(int(self.params.get('epochs', 200)), 1)
        if progress_callback is not None:
            progress_callback(0, total_epochs, {"status": "initializing neural network"})

        from sklearn.neural_network import MLPRegressor
        
        if progress_callback is not None:
            progress_callback(0, total_epochs, {"status": "preparing normalized training data"})

        input_dim = self.X_train.shape[1]
        self.x_mean = np.nanmean(self.X_train, axis=0)
        self.x_std = np.nanstd(self.X_train, axis=0)
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = np.nanmean(self.Y_train, axis=0)
        self.y_std = np.nanstd(self.Y_train, axis=0)
        self.y_std[self.y_std == 0] = 1.0

        X_train = (self.X_train - self.x_mean) / self.x_std
        X_eval = (self.X_eval - self.x_mean) / self.x_std
        Y_train = (self.Y_train - self.y_mean) / self.y_std
        Y_eval = (self.Y_eval - self.y_mean) / self.y_std

        batch_size = int(self.params.get('batch_size', 16))
        patience = int(self.params.get('early_stopping_rounds', 25))

        hidden_units = int(self.params.get("hidden_units", 128))
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_units, max(hidden_units // 2, 16)),
            activation="relu",
            solver="adam",
            learning_rate_init=float(self.params.get('learning_rate', 0.001)),
            alpha=float(self.params.get("alpha", 0.0001)),
            batch_size=batch_size,
            max_iter=1,
            warm_start=True,
            shuffle=True,
            random_state=42,
        )

        if progress_callback is not None:
            progress_callback(0, total_epochs, {"status": "starting training"})

        best_val_loss = np.inf
        best_params = None
        wait = 0

        for epoch in range(total_epochs):
            if progress_callback is not None and epoch == 0:
                progress_callback(0.01, total_epochs, {"status": "running first epoch"})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.model.fit(X_train, Y_train)
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_eval)
            train_loss = float(np.mean(np.abs(train_pred - Y_train)))
            val_loss = float(np.mean(np.abs(val_pred - Y_eval)))

            if progress_callback is not None:
                progress_callback(epoch + 1, total_epochs, {"loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = (
                    [coef.copy() for coef in self.model.coefs_],
                    [intercept.copy() for intercept in self.model.intercepts_],
                )
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_params is not None:
            self.model.coefs_ = best_params[0]
            self.model.intercepts_ = best_params[1]

        if progress_callback is not None:
            progress_callback(epoch + 1, total_epochs, {"val_loss": best_val_loss})

    def evaluateModel(self):
        X_test = (self.X_test - self.x_mean) / self.x_std
        self.preds = (self.model.predict(X_test) * self.y_std) + self.y_mean
    
    def make_cdf_figure(self, offsetRA, errorRA, offsetDec, errorDec, eps=1e-8, nbins=512):
        def to_logabs(x):
            x = np.asarray(x)
            y = np.log10(np.abs(x) + eps)
            return y[np.isfinite(y)]

        def common_bin_edges(a, b, nbins):
            xmin = min(np.min(a), np.min(b))
            xmax = min(np.max(a), np.max(b))
            if xmax <= xmin:
                xmax = xmin + 1e-9
            return np.linspace(xmin, xmax, nbins + 1)

        la_off_RA  = to_logabs(offsetRA)
        la_err_RA  = to_logabs(errorRA)
        la_off_Dec = to_logabs(offsetDec)
        la_err_Dec = to_logabs(errorDec)

        edges_RA  = common_bin_edges(la_off_RA,  la_err_RA,  nbins)
        edges_Dec = common_bin_edges(la_off_Dec, la_err_Dec, nbins)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

        ax1.hist(la_off_RA, bins=edges_RA, density=True, cumulative=True,
                    histtype="step", label="Offset RA")
        ax1.hist(la_err_RA, bins=edges_RA, density=True, cumulative=True,
                    histtype="step", label="Error in Predicting Offset RA")
        ax1.set_xlim(edges_RA[0], edges_RA[-1])
        ax1.set_title("CDF of Log Absolute Error in RA Offset")
        ax1.set_xlabel("Log10 Absolute Error (Degrees)")

        ax2.hist(la_off_Dec, bins=edges_Dec, density=True, cumulative=True,
                    histtype="step", label="Offset Dec")
        ax2.hist(la_err_Dec, bins=edges_Dec, density=True, cumulative=True,
                    histtype="step", label="Error in Predicting Offset Dec")
        ax2.set_xlim(edges_Dec[0], edges_Dec[-1])
        ax2.set_title("CDF of Log Absolute Error in Dec Offset")
        ax2.set_xlabel("Log10 Absolute Error (Degrees)")

        for ax in (ax1, ax2):
            ax.set_ylim(0, 1)
            ax.set_ylabel("Percentage of Points")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, linestyle="--", alpha=0.25)

        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        return fig
    
    def createCDF(self):
        offsetMagnitude = np.abs(self.Y_test)
        error = self.preds - self.Y_test
        offsetRA, offsetDec = offsetMagnitude[:,0], offsetMagnitude[:,1]
        errorRA, errorDec = error[:,0], error[:,1]
        eps = 1e-10

        return self.make_cdf_figure(offsetRA, errorRA, offsetDec, errorDec, eps)
    
    def displayStatistics(self):
        offsetMagnitude = np.abs(self.Y_test)
        error = self.preds - self.Y_test
        offsetRA, offsetDec = offsetMagnitude[:,0], offsetMagnitude[:,1]
        errorRA, errorDec = error[:,0], error[:,1]

        return offsetRA, offsetDec, errorRA, errorDec
    
    '''
    Features needed depend on input file
    STILL NEED TO MAKE THIS WORK
    '''
    def formatPrediction(self, mountRA, mountDec, 
                         second, minute, hour, day, month, year,
                         currList, extraOn, feature_values=None):
        feature_values = feature_values or {}
        n_base_cols = 10 + len(self.custom_feature_names)
        full_row = np.zeros(n_base_cols + (4 if extraOn else 0), dtype=float)
        full_row[:10] = [year, month, day, hour, minute, second, mountRA, mountDec, 0, 0]

        for offset, name in enumerate(self.custom_feature_names, start=10):
            full_row[offset] = float(feature_values.get(name, self.feature_defaults.get(name, 0.0)))

        if extraOn:
            if len(currList) > 0:
                previous_acq_error_RA, previous_acq_error_Dec, previous_acq_RA, previous_acq_Dec = currList[-1]
            else:
                previous_acq_error_RA, previous_acq_error_Dec, previous_acq_RA, previous_acq_Dec = 0, 0, 0, 0
            full_row[-4:] = [previous_acq_error_RA, previous_acq_error_Dec, previous_acq_RA, previous_acq_Dec]

        if self.x_idx is None:
            raise RuntimeError("Model has not been split/trained yet.")
        return np.asarray([full_row[self.x_idx]])

    def updatePastList(self, currList, inputRA, inputDec, actualRA, actualDec):
        previous_acq_error_RA, previous_acq_error_Dec = inputRA - actualRA, inputDec - actualDec
        previous_acq_RA, previous_acq_Dec = inputRA, inputDec
        currList.append((previous_acq_error_RA, previous_acq_error_Dec, previous_acq_RA, previous_acq_Dec))


    def makePrediction(self, mountRA, mountDec, 
                         second, minute, hour, day, month, year,
                         currList, extraOn, feature_values=None):
        if self.model:

            input_pred = self.formatPrediction(mountRA, mountDec,
                                second, minute, hour, day, month, year,
                                currList, extraOn, feature_values)
            
            input_pred = (input_pred - self.x_mean) / self.x_std
            outs = (self.model.predict(input_pred) * self.y_std) + self.y_mean
            return outs
        else:
            print("No Model Trained")
            return None
