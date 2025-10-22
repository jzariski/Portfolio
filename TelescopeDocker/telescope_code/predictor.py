# telescope_code/predictor.py
# Loads the saved models and predicts offsets.
# Accepts either raw string inputs (DATE-OBS, LST, TPT-RA, TPT-DEC) or
# numeric features (years, months, days, hours, minutes, seconds, lst_hours, tpt_ra_deg, tpt_dec_deg).
# Maintains per-session memory (prev_ra_offset, prev_dec_offset).

from __future__ import annotations
import os, json, re
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from .trainer import hms_to_hours, ra_hms_to_deg, dms_to_deg

# 
# Loading
# 

def load_model(model_dir: str):
    """
    Load ra/dec models and meta.json from a model_dir.
    Returns (models_dict, meta_dict) or (None, None) on failure.
    """
    try:
        ra_path = os.path.join(model_dir, "ra_model.joblib")
        dec_path = os.path.join(model_dir, "dec_model.joblib")
        meta_path = os.path.join(model_dir, "meta.json")

        models = {
            "ra": joblib.load(ra_path),
            "dec": joblib.load(dec_path),
        }
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return models, meta
    except Exception as e:
        print(f"[load_model] Failed: {e}")
        return None, None

# 
# Input parsing and feature building
# 

NUMERIC_KEYS = [
    "years","months","days","hours","minutes","seconds",
    "lst_hours","tpt_ra_deg","tpt_dec_deg"
]

def _parse_one_row(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Accepts either:
      - strings: DATE-OBS, LST, TPT-RA, TPT-DEC
      - or numeric pre-parsed features: in NUMERIC_KEYS above
    Returns a dict with NUMERIC_KEYS populated.
    """
    out: Dict[str, float] = {}

    # If numeric version is already provided, just take those.
    if all(k in row for k in NUMERIC_KEYS):
        for k in NUMERIC_KEYS:
            out[k] = float(row[k])
        return out

    # Otherwise parse from strings. DATE-OBS can be ISO; we extract components.
    if not all(k in row for k in ["DATE-OBS","LST","TPT-RA","TPT-DEC"]):
        raise ValueError("Provide either numeric features or the string fields: DATE-OBS, LST, TPT-RA, TPT-DEC.")

    # DATE-OBS
    ts = pd.to_datetime(row["DATE-OBS"], errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid DATE-OBS: {row['DATE-OBS']!r}")

    out["years"]   = float(ts.year)
    out["months"]  = float(ts.month)
    out["days"]    = float(ts.day)
    out["hours"]   = float(ts.hour)
    out["minutes"] = float(ts.minute)
    out["seconds"] = float(ts.second)

    # LST (hms -> hours float)
    out["lst_hours"] = float(hms_to_hours(str(row["LST"])))

    # TPT-RA (hms -> degrees), TPT-DEC (dms -> degrees)
    out["tpt_ra_deg"]  = float(ra_hms_to_deg(str(row["TPT-RA"])))
    out["tpt_dec_deg"] = float(dms_to_deg(str(row["TPT-DEC"])))

    return out


def build_features_from_instances(instances: List[Dict[str, Any]]):
    """
    Build a DataFrame of numeric features from mixed inputs.
    Returns (df, parse_info) where parse_info echoes how many rows were parsed from strings vs numeric.
    """
    numeric_rows = []
    parsed_from_strings = 0
    from_numeric = 0

    for row in instances:
        try:
            nr = _parse_one_row(row)
            # count how we parsed:
            if all(k in row for k in NUMERIC_KEYS):
                from_numeric += 1
            else:
                parsed_from_strings += 1
            numeric_rows.append(nr)
        except Exception as e:
            raise ValueError(f"Error parsing row {row}: {e}")

    df = pd.DataFrame(numeric_rows, columns=NUMERIC_KEYS)
    info = {
        "rows": len(instances),
        "parsed_from_strings": parsed_from_strings,
        "accepted_numeric": from_numeric
    }
    return df, info

# 
# Prediction
# 

def predict_df(
    feat_df: pd.DataFrame,
    session_state: Dict[str, float],
    models: Dict[str, Any],
    meta: Dict[str, Any]
):
    """
    Predict ra/dec offsets for the rows in feat_df using:
      X = base features + prev_ra_offset/prev_dec_offset (from session_state)
    Updates and returns the new session_state after processing the batch.

    Returns (predictions_list, new_session_state_dict)
    """
    # Assemble feature matrix in the exact order used during training
    feature_cols = meta["feature_cols"]
    X = feat_df.copy()

    # Add prev_* from session
    prev_ra = float(session_state.get("prev_ra_offset", 0.0))
    prev_dec = float(session_state.get("prev_dec_offset", 0.0))
    X["prev_ra_offset"] = prev_ra
    X["prev_dec_offset"] = prev_dec

    # Make sure all required feature columns exist and are ordered
    X = X.reindex(columns=feature_cols)

    # Predict with both models
    y_ra  = models["ra"].predict(X.values)
    y_dec = models["dec"].predict(X.values)

    # Update session memory to the LAST prediction in this batch
    new_state = {
        "prev_ra_offset": float(y_ra[-1]),
        "prev_dec_offset": float(y_dec[-1])
    }

    preds = [{"ra_offset": float(r), "dec_offset": float(d)} for r, d in zip(y_ra, y_dec)]
    return preds, new_state
