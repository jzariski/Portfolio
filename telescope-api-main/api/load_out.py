#!/usr/bin/env python3

## Hi
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime, timezone
from astropy.time import Time
from astropy.coordinates import SkyCoord, CIRS
import astropy.units as u
import argparse


# Enable/disable time-gap filtering and threshold (seconds)
# Enable/disable similarity filtering and parameters:
# keep only the first SIM_K acquisitions whose TPT-RA/Dec change is < EPSILON
# Which row index to sample for verbose inspection
SAMPLE_INDEX = 0
# Output filenames
OUTPUT_H5 = 'data/data.h5'


# python load_out.py --input 'data/2025_05_09.txt' --dt 60 --toCIRS 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--dt", type=float, default=0.0)
    parser.add_argument("--toCIRS", action="store_true")
    parser.add_argument("--sim", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.1)

    return parser.parse_args()

def parse_iso_datetime(dt_str):
    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
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
    else:
        # Create SkyCoord for all rows
        coords_icrs = SkyCoord(ra=arr[:,9]*u.deg,
                               dec=arr[:,10]*u.deg,
                               frame='icrs')
    
        years, months, days, hours, minutes, seconds = [arr[:,i].astype(int) for i in range(6)]
    
        times = Time(
            {
                'year'  : years,
                'month' : months,
                'day'   : days,
                'hour'  : hours,
                'minute': minutes,
                'second': seconds
            },
            format='ymdhms',
            scale='utc'
        )
        # Transform to apparent (CIRS)
        coords_app = coords_icrs.transform_to(CIRS(obstime=times))
    
        # Add apparent coordinates to DataFrame
        ra_app = coords_app.ra.deg
        dec_app = coords_app.dec.deg
    
        newArr[:,9] = ra_app
        newArr[:,10] = dec_app
    
        return newArr

def load_out_file(filename):
    raw_lines = []  # keep for sample inspection
    cols = { 'years':[], 'months':[], 'days':[],
             'hours':[], 'minutes':[], 'seconds':[],
             'lst':[], 'tpt_ra':[], 'tpt_dec':[], 'wcs_ra':[], 'wcs_dec':[] }

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # skip blank and header lines
            if not line or line.startswith(('1m0a','Longitude','DATE-OBS')):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            raw_lines.append(line)
            dt_str, lst_str, ra_str, dec_str, wcs_ra_str, wcs_dec_str = parts
            dt = parse_iso_datetime(dt_str)
            cols['years'].append(dt.year)
            cols['months'].append(dt.month)
            cols['days'].append(dt.day)
            cols['hours'].append(dt.hour)
            cols['minutes'].append(dt.minute)
            cols['seconds'].append(dt.second + dt.microsecond/1e6)
            cols['lst'].append(hms_to_hours(lst_str))
            cols['tpt_ra'].append(hms_to_degrees(ra_str))
            cols['tpt_dec'].append(dms_to_degrees(dec_str))
            cols['wcs_ra'].append(float(wcs_ra_str))
            cols['wcs_dec'].append(float(wcs_dec_str))

    data = np.column_stack([
        cols['years'], cols['months'], cols['days'],
        cols['hours'], cols['minutes'], cols['seconds'],
        cols['lst'], cols['tpt_ra'], cols['tpt_dec'],
        cols['wcs_ra'], cols['wcs_dec']
    ])
    return raw_lines, data


def sort_chrono(raw_lines, data):
    """Sort observations by date/time ascending."""
    # lexsort with keys in reverse order
    idx = np.lexsort(( data[:,5], data[:,4], data[:,3],
                       data[:,2], data[:,1], data[:,0] ))
    return [raw_lines[i] for i in idx], data[idx]


def filter_by_dt(raw_lines, data, min_dt):
    """Remove entries whose Δt from previous is ≤ min_dt seconds."""
    # compute UNIX timestamps
    ts = []
    for r in data:
        y,m,d,h,mi,s = r[:6]
        dt = datetime(int(y),int(m),int(d),int(h),int(mi),int(s), tzinfo=timezone.utc)
        ts.append(dt.timestamp())
    ts = np.array(ts)
    mask = np.ones(len(ts), dtype=bool)
    for i in range(1,len(ts)):
        if ts[i] - ts[i-1] <= min_dt:
            mask[i] = False
    removed = np.count_nonzero(~mask)
    return ([raw_lines[i] for i in range(len(raw_lines)) if mask[i]], data[mask], removed)


def filter_by_similarity(raw_lines, data, K, eps):
    """Keep only first K points with ΔTPT-RA/Dec < eps; drop others."""
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

    INPUT_FILE = args.input
    MIN_DT = args.dt
    ENABLE_TIME_FILTER = False
    if MIN_DT > 0:
        ENABLE_TIME_FILTER = True
    TRANSFORM_TO_CIRS = args.toCIRS

    SIM_K = args.sim
    ENABLE_SIM_FILTER = False
    if SIM_K > 0:
        ENABLE_SIM_FILTER = True
    EPSILON = args.eps

    # 1) Load & parse
    raw, data = load_out_file(INPUT_FILE)
    # 2) Sort
    raw, data = sort_chrono(raw, data)
    print(f"Sorted {len(data)} observations.")
    # 3) Optional filters
    if ENABLE_TIME_FILTER:
        raw, data, rem = filter_by_dt(raw, data, MIN_DT)
        print(f"Time filter removed {rem} rows (Δt ≤ {MIN_DT}s).")
    if ENABLE_SIM_FILTER:
        raw, data, rem = filter_by_similarity(raw, data, SIM_K, EPSILON)
        print(f"Similarity filter removed {rem} rows (beyond first {SIM_K} within ε={EPSILON}°).")
    # 4) Sample inspection
    print(data.shape)
    data = convert(data, TRANSFORM_TO_CIRS)
    print(data.shape)
    # 5) Save outputs
    with h5py.File(OUTPUT_H5, 'w') as hf:
        hf.create_dataset('data', data=data, compression='gzip')
    print(f"Saved HDF5: {OUTPUT_H5}")


# run
if __name__ == '__main__':
    main()