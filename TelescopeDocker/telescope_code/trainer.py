# telescope_code/trainer.py
# Reads a text file with columns:
#   DATE-OBS  LST  TPT-RA  TPT-DEC  CRVAL1  CRVAL2
# Converts strings to numeric features and trains two regressors:
#   - ra_offset  = tpt_ra_deg - CRVAL1
#   - dec_offset = tpt_dec_deg - CRVAL2
#
# We DO NOT wrap RA differences (per your request).
#
# Saves:
#   models/<mdl-YYYYmmdd-HHMMSS[-TAG]>/ra_model.joblib
#   models/<mdl-YYYYmmdd-HHMMSS[-TAG]>/dec_model.joblib
#   models/<...>/meta.json

from __future__ import annotations
import os, io, json, time, re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

_HMS_RE = re.compile(r"^\s*(\d+):(\d+):(\d+(?:\.\d+)?)\s*$")
_DMS_RE = re.compile(r"^\s*([+\-]?)(\d+):(\d+):(\d+(?:\.\d+)?)\s*$")

def hms_to_hours(s: str) -> float:
    """Convert 'HH:MM:SS(.sss)' to fractional hours."""
    m = _HMS_RE.match(s)
    if not m:
        raise ValueError(f"Invalid HMS: {s!r}")
    h = float(m.group(1))
    m_ = float(m.group(2))
    s_ = float(m.group(3))
    return h + m_ / 60.0 + s_ / 3600.0

def ra_hms_to_deg(s: str) -> float:
    """Convert RA 'HH:MM:SS(.sss)' to degrees (hours * 15)."""
    return hms_to_hours(s) * 15.0

def dms_to_deg(s: str) -> float:
    """Convert Dec '±DD:MM:SS(.sss)' to degrees (sign-aware)."""
    m = _DMS_RE.match(s)
    if not m:
        raise ValueError(f"Invalid DMS: {s!r}")
    sign = -1.0 if m.group(1) == "-" else 1.0
    d = float(m.group(2))
    m_ = float(m.group(3))
    s_ = float(m.group(4))
    return sign * (d + m_ / 60.0 + s_ / 3600.0)

# ---------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------

REQ_COLS = ["DATE-OBS", "LST", "TPT-RA", "TPT-DEC", "CRVAL1", "CRVAL2"]

def _read_table_from_text(raw_text: str) -> pd.DataFrame:
    """
    Accepts the entire uploaded file as text.
    Finds the header row that starts with 'DATE-OBS' and loads the table below it.
    """
    lines = [ln.rstrip("\n\r") for ln in raw_text.splitlines()]
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("DATE-OBS"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find a header line beginning with 'DATE-OBS'.")

    txt = "\n".join(lines[header_idx:])  # include header
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+", engine="python")

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def _strings_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string columns to numeric features used for training.
    Produces these numeric columns:
      years, months, days, hours, minutes, seconds,
      lst_hours, tpt_ra_deg, tpt_dec_deg, wcs_ra_deg, wcs_dec_deg
    """
    out = df.copy()

    # Parse DATE-OBS to components
    dt = pd.to_datetime(out["DATE-OBS"], errors="coerce", utc=True)
    if dt.isna().any():
        bad = out.loc[dt.isna(), "DATE-OBS"].head(3).tolist()
        raise ValueError(f"Invalid DATE-OBS values (examples): {bad}")

    out["years"]   = dt.dt.year.astype(int)
    out["months"]  = dt.dt.month.astype(int)
    out["days"]    = dt.dt.day.astype(int)
    out["hours"]   = dt.dt.hour.astype(int)
    out["minutes"] = dt.dt.minute.astype(int)
    # seconds may be fractional; keep as int seconds (drop ns)
    out["seconds"] = dt.dt.second.astype(int)

    # Parse LST / TPT
    out["lst_hours"]   = out["LST"].map(hms_to_hours)
    out["tpt_ra_deg"]  = out["TPT-RA"].map(ra_hms_to_deg)
    out["tpt_dec_deg"] = out["TPT-DEC"].map(dms_to_deg)

    # WCS in degrees provided directly
    out["wcs_ra_deg"]  = pd.to_numeric(out["CRVAL1"], errors="coerce")
    out["wcs_dec_deg"] = pd.to_numeric(out["CRVAL2"], errors="coerce")

    # Compute targets (NO RA wrapping)
    out["ra_offset"]  = out["tpt_ra_deg"]  - out["wcs_ra_deg"]
    out["dec_offset"] = out["tpt_dec_deg"] - out["wcs_dec_deg"]

    # Drop rows with any NaN in features/targets
    needed = ["years","months","days","hours","minutes","seconds",
              "lst_hours","tpt_ra_deg","tpt_dec_deg","ra_offset","dec_offset"]
    out = out.dropna(subset=needed).reset_index(drop=True)
    return out


def _make_xy(df_num: pd.DataFrame):
    """
    Build X (features) and y_ra, y_dec (targets).
    We include prev_* features in the model and set them to 0.0 for training.
    """
    feature_cols = [
        "years","months","days","hours","minutes","seconds",
        "lst_hours","tpt_ra_deg","tpt_dec_deg",
        "prev_ra_offset","prev_dec_offset"
    ]
    X = pd.DataFrame(index=df_num.index)
    X["years"] = df_num["years"]
    X["months"] = df_num["months"]
    X["days"] = df_num["days"]
    X["hours"] = df_num["hours"]
    X["minutes"] = df_num["minutes"]
    X["seconds"] = df_num["seconds"]
    X["lst_hours"] = df_num["lst_hours"]
    X["tpt_ra_deg"] = df_num["tpt_ra_deg"]
    X["tpt_dec_deg"] = df_num["tpt_dec_deg"]
    # during training, previous outputs are unknown → set to 0
    X["prev_ra_offset"] = 0.0
    X["prev_dec_offset"] = 0.0

    y_ra  = df_num["ra_offset"].values
    y_dec = df_num["dec_offset"].values

    return X, y_ra, y_dec, feature_cols


def _make_models():
    """
    Two simple pipelines (scaling + ridge regression).
    You can swap Ridge for another regressor if you like.
    """
    pipe = lambda: Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    return pipe(), pipe()  # (ra_model, dec_model)


def _save_model_bundle(model_root: str, tag: str | None, model_ra, model_dec, meta: Dict[str, Any]) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"mdl-{ts}" + (f"-{tag}" if tag else "")
    model_dir = os.path.join(model_root, base)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model_ra, os.path.join(model_dir, "ra_model.joblib"))
    joblib.dump(model_dec, os.path.join(model_dir, "dec_model.joblib"))

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model_dir


def train_model_from_text(raw_text: str, model_root: str = "models", tag: str | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Main entry point called by the API.
    Returns (model_dir, meta).
    """
    raw_df = _read_table_from_text(raw_text)
    num_df = _strings_to_numeric(raw_df)
    if num_df.empty:
        raise ValueError("After parsing/cleaning, no valid rows remained.")

    X, y_ra, y_dec, feature_cols = _make_xy(num_df)

    model_ra, model_dec = _make_models()
    model_ra.fit(X.values, y_ra)
    model_dec.fit(X.values, y_dec)

    meta = {
        "feature_cols": feature_cols,
        "target_cols": ["ra_offset", "dec_offset"],
        "training_rows": int(len(X)),
        "schema": {
            "train_input": ["DATE-OBS","LST","TPT-RA","TPT-DEC","CRVAL1","CRVAL2"],
            "predict_accepts": [
                # You can send either the raw strings below...
                "DATE-OBS","LST","TPT-RA","TPT-DEC",
                # ...or the already-parsed numeric features below.
                "years","months","days","hours","minutes","seconds",
                "lst_hours","tpt_ra_deg","tpt_dec_deg"
            ],
            "notes": [
                "No RA wrapping applied to (tpt_ra_deg - CRVAL1).",
                "prev_ra_offset/prev_dec_offset are added internally; set to 0 for the first call per session."
            ]
        }
    }

    model_dir = _save_model_bundle(model_root, tag, model_ra, model_dec, meta)
    return model_dir, meta
