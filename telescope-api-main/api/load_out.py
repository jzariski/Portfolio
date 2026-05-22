#!/usr/bin/env python3
"""
load_out.py

Converts raw telescope observation logs into a cleaned HDF5 dataset.
This step parses timestamps, converts coordinate formats, and optionally filters
out data that is too close in time or too similar in pointing.
"""
import numpy as np
import h5py
from datetime import datetime, timezone
from astropy.time import Time
from astropy.coordinates import SkyCoord, CIRS
import astropy.units as u
import argparse
from pathlib import Path


# Parsed rows are stored in this fixed column order. The training and
# prediction code depends on these indices, so keep this list as the source of
# truth when changing the dataset schema.
COLUMN_NAMES = (
    "year", "month", "day",
    "hour", "minute", "second",
    "lst_hours", "obs_ra_deg", "obs_dec_deg",
    "solv_ra_deg", "solv_dec_deg",
)
OUTPUT_H5 = "data/data.h5"


# Example:
# python load_out.py --input data/data.dat --dt 60 --toCIRS


def get_args():
    parser = argparse.ArgumentParser(
        description="Parse raw telescope logs into data/data.h5 for model training."
    )

    parser.add_argument("--input", type=str, required=True, help="Raw whitespace-delimited observation log")
    parser.add_argument("--dt", type=float, default=0.0, help="Minimum seconds between retained rows")
    parser.add_argument("--toCIRS", action="store_true", help="Transform solved RA/Dec from ICRS to CIRS")
    parser.add_argument("--sim", type=int, default=0, help="Keep only the first K consecutive similar pointings")
    parser.add_argument("--eps", type=float, default=0.1, help="Similarity threshold in degrees for --sim")

    return parser.parse_args()

def parse_iso_datetime(dt_str):
    # Some logs include a trailing "Z" for UTC and some do not.
    cleaned = dt_str.rstrip("Z")
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)

def hms_to_hours(hms_str):
    h, m, s = map(float, hms_str.split(':'))
    return h + m/60.0 + s/3600.0

def hms_to_degrees(hms_str):
    return hms_to_hours(hms_str) * 15.0

def dms_to_degrees(dms_str):
    sign = -1 if dms_str.strip().startswith('-') else 1
    d, m, s = map(float, dms_str.lstrip('+-').split(':'))
    return sign * (d + m/60.0 + s/3600.0)
    
def convert(arr, trans):
    newArr = np.copy(arr)
    if not trans:
        return newArr

    # Transform the solved coordinates into the apparent CIRS frame at each
    # observation time. The observed coordinates are left unchanged because the
    # raw log already represents the telescope command/pointing.
    coords_icrs = SkyCoord(
        ra=arr[:, 9] * u.deg,
        dec=arr[:, 10] * u.deg,
        frame="icrs",
    )

    years, months, days, hours, minutes, seconds = [arr[:, i].astype(int) for i in range(6)]
    times = Time(
        {
            "year": years,
            "month": months,
            "day": days,
            "hour": hours,
            "minute": minutes,
            "second": seconds,
        },
        format="ymdhms",
        scale="utc",
    )
    coords_app = coords_icrs.transform_to(CIRS(obstime=times))

    newArr[:, 9] = coords_app.ra.deg
    newArr[:, 10] = coords_app.dec.deg
    return newArr

def load_out_file(filename):
    raw_lines = []  # Keep original lines so filtered-row reporting stays traceable.
    cols = { 'years':[], 'months':[], 'days':[],
             'hours':[], 'minutes':[], 'seconds':[],
             'lst':[], 'obs_ra':[], 'obs_dec':[], 'solv_ra':[], 'solv_dec':[] }

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank/header lines and any malformed rows instead of failing
            # the entire import on one bad acquisition record.
            if not line or line.startswith(('1m0a','Longitude','DATE-OBS')):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            raw_lines.append(line)
            dt_str, lst_str, ra_str, dec_str, solv_ra_str, solv_dec_str = parts
            dt = parse_iso_datetime(dt_str)
            cols['years'].append(dt.year)
            cols['months'].append(dt.month)
            cols['days'].append(dt.day)
            cols['hours'].append(dt.hour)
            cols['minutes'].append(dt.minute)
            cols['seconds'].append(dt.second + dt.microsecond/1e6)
            cols['lst'].append(hms_to_hours(lst_str))
            cols['obs_ra'].append(hms_to_degrees(ra_str))
            cols['obs_dec'].append(dms_to_degrees(dec_str))
            cols['solv_ra'].append(float(solv_ra_str))
            cols['solv_dec'].append(float(solv_dec_str))

    data = np.column_stack([
        cols['years'], cols['months'], cols['days'],
        cols['hours'], cols['minutes'], cols['seconds'],
        cols['lst'], cols['obs_ra'], cols['obs_dec'],
        cols['solv_ra'], cols['solv_dec']
    ])
    return raw_lines, data


def sort_chrono(raw_lines, data):
    """Sort observations by date/time ascending."""
    # np.lexsort receives keys from least significant to most significant.
    idx = np.lexsort(( data[:,5], data[:,4], data[:,3],
                       data[:,2], data[:,1], data[:,0] ))
    return [raw_lines[i] for i in idx], data[idx]


def filter_by_dt(raw_lines, data, min_dt):
    """Remove entries whose time since the previous row is <= min_dt seconds."""
    # Build timestamps with fractional seconds preserved; these are sorted
    # already, so a vectorized diff is enough.
    ts = np.array([
        datetime(
            int(y),
            int(m),
            int(d),
            int(h),
            int(mi),
            int(s),
            int((s % 1) * 1_000_000),
            tzinfo=timezone.utc,
        ).timestamp()
        for y, m, d, h, mi, s in data[:, :6]
    ])
    mask = np.ones(len(ts), dtype=bool)
    mask[1:] = np.diff(ts) > min_dt
    removed = np.count_nonzero(~mask)
    return ([raw_lines[i] for i in range(len(raw_lines)) if mask[i]], data[mask], removed)


def filter_by_similarity(raw_lines, data, K, eps):
    """Keep only the first K consecutive rows with similar observed RA/Dec."""
    keep = [True]
    sim_count = 0
    for i in range(1,len(data)):
        dra = abs(data[i,7] - data[i-1,7])
        ddec= abs(data[i,8] - data[i-1,8])
        if dra < eps and ddec < eps:
            sim_count += 1
            keep.append(sim_count <= K)
        else:
            keep.append(True)
    mask = np.array(keep)
    removed = np.count_nonzero(~mask)
    return ([raw_lines[i] for i in range(len(raw_lines)) if mask[i]], data[mask], removed)



def main():
    args = get_args()

    enable_time_filter = args.dt > 0
    enable_sim_filter = args.sim > 0

    # 1) Load & parse
    raw, data = load_out_file(args.input)
    # 2) Sort
    raw, data = sort_chrono(raw, data)
    print(f"Sorted {len(data)} observations.")
    # 3) Optional filters
    if enable_time_filter:
        raw, data, rem = filter_by_dt(raw, data, args.dt)
        print(f"Time filter removed {rem} rows (delta_t <= {args.dt}s).")
    if enable_sim_filter:
        raw, data, rem = filter_by_similarity(raw, data, args.sim, args.eps)
        print(f"Similarity filter removed {rem} rows (beyond first {args.sim} within eps={args.eps} deg).")

    # 4) Optional coordinate-frame conversion
    data = convert(data, args.toCIRS)
    print(f"Final dataset shape: {data.shape} columns={COLUMN_NAMES}")

    # 5) Save outputs
    Path(OUTPUT_H5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_H5, 'w') as hf:
        hf.create_dataset('data', data=data, compression='gzip')
    print(f"Saved HDF5: {OUTPUT_H5}")


# run
if __name__ == '__main__':
    main()
