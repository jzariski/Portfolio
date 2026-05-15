# FLIPoint

FLIPoint is a Streamlit application for learning telescope acquisition pointing
corrections from previous observations. It trains a model on historical mount
coordinates, actual acquired coordinates, timestamps, and optional observing
features, then predicts the RA/Dec offset to apply to a new requested mount
position.

The app is designed for an acquisition loop:

1. Upload an observation log.
2. Train an XGBoost or neural-network correction model.
3. Evaluate correction quality on held-out observations.
4. Predict the next acquisition offset.
5. Add the actual acquired position back into the live prediction history so
   autoregressive features can improve the next prediction.

## What FLIPoint Predicts

For each observation, FLIPoint learns the offset:

```text
RA offset  = MountRA  - ActualRA
Dec offset = MountDec - ActualDec
```

When you enter a new requested `MountRA` and `MountDec`, the app predicts those
offsets and displays the corrected coordinates:

```text
Corrected RA  = MountRA  - predicted RA offset
Corrected Dec = MountDec - predicted Dec offset
```

If additional feature engineering is enabled, FLIPoint also uses information
from the previous acquisition, including the previous RA/Dec acquisition error
and previous mount position.

## Installation

This project uses Python and Streamlit. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the App

Start the Streamlit UI with:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Input File Format

Upload a `.txt` observation file in the sidebar. The file must begin with a
site-coordinate line, followed by a column-header line, followed by whitespace
separated observation rows.

Required first line:

```text
Longitude = -110.95deg Latitude = 32.18deg
```

Required columns:

```text
Date Time MountRA MountDec ActualRA ActualDec
```

Additional columns are optional. Numeric columns and boolean values
(`True`/`False`) are parsed automatically and used as model features.

Example:

```text
Longitude = -110.95deg Latitude = 32.18deg
Date Time MountRA MountDec ActualRA ActualDec Pressure Count Temperature
2021-01-01 00:00:00 188.1 44.7 188.3 44.7 48.6 True 11
2021-01-01 01:00:00 176.9 40.1 177.0 40.2 51.42 False 27
```

The Streamlit app currently parses `Date` and `Time` as `YYYY-MM-DD` and
`HH:MM:SS`. Small example files are included at `ymd_example_small.txt` and
`Parsing/ymd_example.txt`.

## Using FLIPoint

### 1. Upload Data

Use the sidebar file uploader to select a `.txt` observation file. After a
successful parse, the app shows the number of observations, parsed features,
longitude, latitude, and a preview of the uploaded data.

### 2. Configure Training

Choose a model family:

- `XGBoost`: gradient-boosted tree regression with controls for estimators,
  max depth, learning rate, and early stopping.
- `Neural network`: scikit-learn MLP regression with controls for epochs, batch
  size, hidden units, learning rate, L2 regularization, and patience.

You can also configure:

- Whether to enable additional autoregressive feature engineering.
- Whether to split train/validation/test data by date.
- Train, validation, and test percentages.
- Whether to show summary statistics and CDF plots after training.

Click `Train Model` when the settings are ready.

### 3. Evaluate the Model

After training, open the `Evaluate` tab. FLIPoint evaluates the model on the
held-out test set and reports:

- Median and selected-percentile RA/Dec offsets.
- Median and selected-percentile prediction errors.
- Mean, standard deviation, count, median, and percentile tables.
- CDF plots comparing observed offsets with prediction errors.

Use these metrics to decide whether the trained correction model is accurate
enough for your acquisition workflow.

### 4. Predict and Augment

Open the `Predict & Augment` tab and enter:

- Requested `MountRA` and `MountDec`.
- Observation timestamp.
- Any extra feature values that were present in the uploaded file.

Click `Predict Offset` to get the predicted RA/Dec offset and corrected
coordinates.

If additional feature engineering is enabled, FLIPoint then asks for the actual
acquired `ActualRA` and `ActualDec`. Click `Add Observation` to add that result
to the live autoregressive history used by the next prediction.

### 5. Review History

The `History` tab shows:

- Previous predicted RA/Dec offsets.
- Previous actual acquisition errors and mount positions added during the
  predict-and-augment cycle.

## Smoke Test

To verify that parsing, training, evaluation, and prediction work outside the
Streamlit UI, run:

```bash
python smoke.py
```

The script trains a small XGBoost model using `ymd_example_small.txt` and prints
the parsed row count, input features, and one prediction.

## Project Layout

```text
app.py                    Streamlit application
Parsing/TextParser.py     Observation text-file parser
ModelBuilder/ModelBuilder.py
                          XGBoost correction model
ModelBuilder/ModelBuilderNN.py
                          Neural-network correction model
smoke.py                  Minimal end-to-end smoke test
requirements.txt          Python dependencies
```

## Notes

- Train, validation, and test splits must all contain at least one observation.
  If training fails with a split error, use more data or adjust the percentages.
- Extra feature columns should be numeric or boolean. Non-numeric strings will
  fail parsing.
- Coordinate units are displayed using the `Coordinate units` sidebar value. The
  example data uses degrees.
