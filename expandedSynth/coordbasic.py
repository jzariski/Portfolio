import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"] = 18

from datetime import datetime, timezone

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.utils import iers

import xgboost as xgb



OUT_PATH = "2024_10_2025_09.out"   # .out file

# Telescope location
TELESCOPE_LAT_DEG = -32.383
TELESCOPE_LON_DEG = 20.813    # positive = East, negative = West
TELESCOPE_HEIGHT_M = 1760

WCS_FRAME = "icrs"
RNG_SEED = 7
PLOT_OUTLIER_PERCENTILE = 99.0

# XGBoost settings
## Expanded but can be altered
## Feature engineering opportunites can be done here with grid search, bayesian, etc.
XGB_PARAMS = dict(
    booster="gbtree",
    objective="reg:squarederror",
    eval_metric="rmse",

    n_estimators=5000,
    learning_rate=0.02,

    max_depth=5,
    min_child_weight=2,

    subsample=0.95,
    colsample_bytree=0.95,

    reg_lambda=2.0,
    reg_alpha=0.0,
    gamma=0.0,

    tree_method="hist",
    max_bin=1024,

    random_state=RNG_SEED,
    n_jobs=-1,
    early_stopping_rounds=150,
)

EARLY_STOPPING_ROUNDS = 150



def hms_to_hours(hms_str: str) -> float:
    h, m, s = map(float, hms_str.split(":"))
    return h + m / 60.0 + s / 3600.0


def hms_to_degrees(hms_str: str) -> float:
    return 15.0 * hms_to_hours(hms_str)


def dms_to_degrees(dms_str: str) -> float:
    s = dms_str.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    d, m, sec = map(float, s.split(":"))
    return sign * (d + m / 60.0 + sec / 3600.0)


def parse_dt_utc(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_array_from_out(path_out: str) -> np.ndarray:
    """
    Returns (N,11) array:
      0..5: year, month, day, hour, minute, second
      6:    LST (hours)
      7..8: TPT/OBS RA/Dec (deg)
      9..10:WCS RA/Dec (deg)
    """
    years, months, days = [], [], []
    hours, minutes, seconds = [], [], []
    lst_hours = []
    obs_ra, obs_dec = [], []
    wcs_ra, wcs_dec = [], []

    with open(path_out, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Longitude") or line.startswith("DATE-OBS") or line.startswith("#"):
                continue
            # first line sometimes is a filename — skip it
            if line.endswith(".cpt") and " " not in line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            dt_str, lst_str, tpt_ra_str, tpt_dec_str, crval1_str, crval2_str = parts[:6]
            dt = parse_dt_utc(dt_str)

            years.append(dt.year)
            months.append(dt.month)
            days.append(dt.day)
            hours.append(dt.hour)
            minutes.append(dt.minute)
            seconds.append(dt.second + dt.microsecond / 1e6)

            lst_hours.append(hms_to_hours(lst_str))
            obs_ra.append(hms_to_degrees(tpt_ra_str))
            obs_dec.append(dms_to_degrees(tpt_dec_str))

            wcs_ra.append(float(crval1_str))
            wcs_dec.append(float(crval2_str))

    return np.column_stack(
        [
            np.array(years),
            np.array(months),
            np.array(days),
            np.array(hours),
            np.array(minutes),
            np.array(seconds),
            np.array(lst_hours),
            np.array(obs_ra),
            np.array(obs_dec),
            np.array(wcs_ra),
            np.array(wcs_dec),
        ]
    )


def radec_to_horizon(data: np.ndarray, ra_col: int, dec_col: int):
    location = EarthLocation(
        lat=TELESCOPE_LAT_DEG * u.deg,
        lon=TELESCOPE_LON_DEG * u.deg,
        height=TELESCOPE_HEIGHT_M * u.m,
    )

    # Prevent astropy from downloading IERS tables
    iers.conf.auto_download = False

    years = data[:, 0].astype(int)
    months = data[:, 1].astype(int)
    days = data[:, 2].astype(int)
    hours = data[:, 3].astype(int)
    minutes = data[:, 4].astype(int)
    secs = data[:, 5].astype(float)

    times = Time(
        {
            "year": years,
            "month": months,
            "day": days,
            "hour": hours,
            "minute": minutes,
            "second": secs,
        },
        format="ymdhms",
        scale="utc",
    )

    sky = SkyCoord(
        ra=data[:, ra_col].astype(float) * u.deg,
        dec=data[:, dec_col].astype(float) * u.deg,
        frame=WCS_FRAME,
    )

    altaz_frame = AltAz(
        obstime=times,
        location=location,
        pressure=0 * u.hPa,   # no refraction
    )
    altaz = sky.transform_to(altaz_frame)

    return altaz.az.deg, altaz.alt.deg


## Appends the array with transformed celestial to hroizon targets
def add_horizon_targets(data: np.ndarray) -> np.ndarray:
    """
    Appends two columns:
      11: wcs_az (deg)
      12: wcs_el (deg)
    """
    wcs_az, wcs_el = radec_to_horizon(data, ra_col=9, dec_col=10)

    return np.hstack((data, wcs_az[:, None], wcs_el[:, None]))


## Angle wrapping to account for circular natures
def wrap_angle_deg(angle_deg):
    """Wrap angle to [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def wrap_angle_error_deg(pred_deg, true_deg):
    """Smallest signed difference in degrees, in [-180, 180)."""
    return ((pred_deg - true_deg + 180.0) % 360.0) - 180.0


def make_features_and_targets(data: np.ndarray):
    """
    Features use only:
      - LST from file
      - WCS RA/Dec from file

    Derived feature:
      - hour angle = LST - RA

    Targets:
      - Astropy-computed Az/El
    """
    lst_hours = data[:, 6].astype(float)
    wcs_ra_deg = data[:, 9].astype(float)
    wcs_dec_deg = data[:, 10].astype(float)

    # Convert LST hours -> degrees
    lst_deg = (15.0 * lst_hours) % 360.0

    # Hour angle in degrees, wrapped
    ha_deg = wrap_angle_deg(lst_deg - wcs_ra_deg)

    lst_rad = np.deg2rad(lst_deg)
    ra_rad = np.deg2rad(wcs_ra_deg)
    ha_rad = np.deg2rad(ha_deg)
    dec_rad = np.deg2rad(wcs_dec_deg)
    lat_rad = np.deg2rad(TELESCOPE_LAT_DEG)

    # Low-order geometric horizon transform features. These are not Astropy
    # targets, but they give the booster a physically shaped starting point.
    sin_el_geom = (
        np.sin(dec_rad) * np.sin(lat_rad)
        + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad)
    )
    sin_el_geom = np.clip(sin_el_geom, -1.0, 1.0)
    el_geom_rad = np.arcsin(sin_el_geom)
    cos_el_geom = np.maximum(np.cos(el_geom_rad), 1e-12)
    sin_az_geom = -np.sin(ha_rad) * np.cos(dec_rad) / cos_el_geom
    cos_az_geom = (
        np.sin(dec_rad) - np.sin(el_geom_rad) * np.sin(lat_rad)
    ) / (cos_el_geom * np.cos(lat_rad))

    # ADDITIONAL FEATURE ENGINEERING
    # CAN APPEND THIS WITH OTHER FEATURES IF YOU WANT
    X = np.column_stack([
        np.sin(lst_rad),
        np.cos(lst_rad),
        np.sin(ra_rad),
        np.cos(ra_rad),
        np.sin(ha_rad),
        np.cos(ha_rad),
        np.sin(dec_rad),
        np.cos(dec_rad),
        ha_deg / 180.0,
        wcs_dec_deg / 90.0,
        sin_el_geom,
        np.cos(el_geom_rad),
        sin_az_geom,
        cos_az_geom,
    ])

    az_deg = data[:, 11].astype(float)
    el_deg = data[:, 12].astype(float)

    return X, az_deg, el_deg


## Makes splits. Note NON-TEMPORAL (This should be changed for real system use)
def make_split_indices(n, seed=7, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)

    n_train = int(0.70 * n)
    n_eval = int(0.10 * n)

    train_idx = idx[:n_train]
    eval_idx = idx[n_train:n_train + n_eval]
    test_idx = idx[n_train + n_eval:]

    return train_idx, eval_idx, test_idx


def split_with_indices(arr, train_idx, eval_idx, test_idx):
    return arr[train_idx], arr[eval_idx], arr[test_idx]



# Load data and build Astropy targets
data = load_array_from_out(OUT_PATH)
data = add_horizon_targets(data)

X, true_az_deg_all, true_el_deg_all = make_features_and_targets(data)

# Predict Az as sin/cos to avoid wrap issues
az_rad_all = np.deg2rad(true_az_deg_all)
y_az_sin_all = np.sin(az_rad_all)
y_az_cos_all = np.cos(az_rad_all)

# Predict El directly
y_el_all = true_el_deg_all.copy()

# Train/eval/test split
train_idx, eval_idx, test_idx = make_split_indices(
    n=X.shape[0],
    seed=RNG_SEED,
    shuffle=True,
)

X_train, X_eval, X_test = split_with_indices(X, train_idx, eval_idx, test_idx)

y_az_sin_train, y_az_sin_eval, y_az_sin_test = split_with_indices(
    y_az_sin_all, train_idx, eval_idx, test_idx
)
y_az_cos_train, y_az_cos_eval, y_az_cos_test = split_with_indices(
    y_az_cos_all, train_idx, eval_idx, test_idx
)
y_el_train, y_el_eval, y_el_test = split_with_indices(
    y_el_all, train_idx, eval_idx, test_idx
)

true_az_test = true_az_deg_all[test_idx]
true_el_test = true_el_deg_all[test_idx]

print("Shapes:")
print("  X_train:", X_train.shape)
print("  X_eval :", X_eval.shape)
print("  X_test :", X_test.shape)

# Train models: Train separate models for trig components of RA, reconstruct later
model_sin = xgb.XGBRegressor(**XGB_PARAMS)
model_cos = xgb.XGBRegressor(**XGB_PARAMS)
model_el = xgb.XGBRegressor(**XGB_PARAMS)

model_sin.fit(
    X_train, y_az_sin_train,
    eval_set=[(X_eval, y_az_sin_eval)],
    verbose=False,
)

model_cos.fit(
    X_train, y_az_cos_train,
    eval_set=[(X_eval, y_az_cos_eval)],
    verbose=False,
)

model_el.fit(
    X_train, y_el_train,
    eval_set=[(X_eval, y_el_eval)],
    verbose=False,
)

# Predict on test set
pred_sin = model_sin.predict(X_test)
pred_cos = model_cos.predict(X_test)

# Normalize sin/cos predictions before atan2
r = np.sqrt(pred_sin**2 + pred_cos**2) + 1e-12
pred_sin = pred_sin / r
pred_cos = pred_cos / r

# Return to Euclidean coordinates
pred_az_deg = (np.rad2deg(np.arctan2(pred_sin, pred_cos)) + 360.0) % 360.0
pred_el_deg = np.clip(model_el.predict(X_test), -90.0, 90.0)

# Errors
az_err = wrap_angle_error_deg(pred_az_deg, true_az_test)
el_err = pred_el_deg - true_el_test

# Original pointing offsets from TPT/OBS coordinates to WCS coordinates,
# converted into Az/El so they are directly comparable to model errors.
obs_az_all, obs_el_all = radec_to_horizon(data, ra_col=7, dec_col=8)
obs_az_test = obs_az_all[test_idx]
obs_el_test = obs_el_all[test_idx]
orig_az_offset = wrap_angle_error_deg(obs_az_test, true_az_test)
orig_el_offset = obs_el_test - true_el_test

print("\nTest errors:")
print("  Az MAE (wrapped deg):", np.mean(np.abs(az_err)))
print("  Az RMSE (wrapped deg):", np.sqrt(np.mean(az_err**2)))
print("  Az abs Med (wrapped deg):", np.median(np.abs(az_err)))

print("  El MAE (deg):", np.mean(np.abs(el_err)))
print("  El RMSE (deg):", np.sqrt(np.mean(el_err**2)))
print("  El abs Med (deg):", np.median(np.abs(el_err)))

print("\nOriginal offset magnitudes:")
print("  Az mean abs offset (wrapped deg):", np.mean(np.abs(orig_az_offset)))
print("  Az median abs offset (wrapped deg):", np.median(np.abs(orig_az_offset)))
print("  El mean abs offset (deg):", np.mean(np.abs(orig_el_offset)))
print("  El median abs offset (deg):", np.median(np.abs(orig_el_offset)))

## Various types of plots

# True vs predicted Az
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
ax.scatter(true_az_test, pred_az_deg, s=10, alpha=0.35)
ax.set_xlabel("True Az (deg) [astropy]")
ax.set_ylabel("Pred Az (deg) [xgb]")
ax.set_title("Az: True vs Pred")
ax.grid(False)
plt.tight_layout()
plt.savefig("azplot.png", dpi=200)

# True vs predicted El
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
ax.scatter(true_el_test, pred_el_deg, s=10, alpha=0.35)
ax.set_xlabel("True El (deg) [astropy]")
ax.set_ylabel("Pred El (deg) [xgb]")
ax.set_title("El: True vs Pred")
ax.grid(False)
plt.tight_layout()
plt.savefig("elplot.png", dpi=200)

# CDF of absolute errors
az_abs = np.sort(np.abs(az_err))
az_cdf = np.arange(1, az_abs.size + 1) / az_abs.size

el_abs = np.sort(np.abs(el_err))
el_cdf = np.arange(1, el_abs.size + 1) / el_abs.size


# Original offset magnitudes and model error magnitudes by test sample
az_orig_abs = np.abs(orig_az_offset)
az_model_abs = np.abs(az_err)
el_orig_abs = np.abs(orig_el_offset)
el_model_abs = np.abs(el_err)
orig_total_abs = np.sqrt(orig_az_offset**2 + orig_el_offset**2)
model_total_abs = np.sqrt(az_err**2 + el_err**2)

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)

az_keep = az_orig_abs <= np.percentile(az_orig_abs, PLOT_OUTLIER_PERCENTILE)
az_order = np.argsort(az_orig_abs[az_keep])
az_x = np.arange(az_order.size)
axes[0].scatter(az_x, az_orig_abs[az_keep][az_order], s=10, alpha=0.45, label="Original Az offset")
axes[0].scatter(az_x, az_model_abs[az_keep][az_order], s=10, alpha=0.45, label="Model Az error")
axes[0].set_xlabel("Test sample sorted by original Az offset")
axes[0].set_ylabel("Magnitude (deg)")
axes[0].set_title(f"Az Magnitudes (<= {PLOT_OUTLIER_PERCENTILE:g}th percentile original offset)")
axes[0].legend()
axes[0].grid(False)

el_keep = el_orig_abs <= np.percentile(el_orig_abs, PLOT_OUTLIER_PERCENTILE)
el_order = np.argsort(el_orig_abs[el_keep])
el_x = np.arange(el_order.size)
axes[1].scatter(el_x, el_orig_abs[el_keep][el_order], s=10, alpha=0.45, label="Original El offset")
axes[1].scatter(el_x, el_model_abs[el_keep][el_order], s=10, alpha=0.45, label="Model El error")
axes[1].set_xlabel("Test sample sorted by original El offset")
axes[1].set_ylabel("Magnitude (deg)")
axes[1].set_title(f"El Magnitudes (<= {PLOT_OUTLIER_PERCENTILE:g}th percentile original offset)")
axes[1].legend()
axes[1].grid(False)

total_keep = orig_total_abs <= np.percentile(orig_total_abs, PLOT_OUTLIER_PERCENTILE)
total_order = np.argsort(orig_total_abs[total_keep])
total_x = np.arange(total_order.size)
axes[2].scatter(total_x, orig_total_abs[total_keep][total_order], s=10, alpha=0.45, label="Original combined offset")
axes[2].scatter(total_x, model_total_abs[total_keep][total_order], s=10, alpha=0.45, label="Model combined error")
axes[2].set_xlabel("Test sample sorted by original combined offset")
axes[2].set_ylabel("Magnitude (deg)")
axes[2].set_title(f"Combined Az/El Magnitudes (<= {PLOT_OUTLIER_PERCENTILE:g}th percentile original offset)")
axes[2].legend()
axes[2].grid(False)

plt.tight_layout()
plt.savefig("offset_improvement.png", dpi=200)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
ax.hist(np.log10(abs(az_err)), bins=1000, density=True, cumulative=True, histtype='step', label='Error in Predicting Offset RA')
ax.hist(np.log10(abs(el_err)), bins=1000, density=True, cumulative=True, histtype='step', label='Error in Predicting Offset El')
ax.set_xlabel("Log Absolute error (deg)")
ax.set_ylabel("CDF")
ax.set_title("CDF of Absolute Errors")
ax.legend()
ax.grid(False)
plt.tight_layout()
plt.savefig("cdf_true.png", dpi=200)

print("\nSaved:")
print("  azplot.png")
print("  elplot.png")
print("  cdf_errors.png")
print("  offset_improvement.png")
print("  cdf_true.png")

print("\nDone.")
