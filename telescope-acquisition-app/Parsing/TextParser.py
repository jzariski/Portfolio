#!/usr/bin/env python3
"""
parse_to_h5.py

Script to read a text file with a header indicating longitude/latitude,
then a header row of column names, parse date/time and arbitrary data columns,
convert booleans to floats, and save everything to an HDF5 file including
column headers and metadata.

Usage:
    python parse_to_h5.py input.txt output.h5 [--ymd]

By default, the first two columns are assumed to be UNIX timestamps (in column 1)
and ignored in column 2. With --ymd, first column must be YYYY-MM-DD and the
second column must be HH:MM:SS.

Important to note that there must be a header line, with first headings always containing:
Date Time MountAz MountAlt ActualAz ActualAlt
Following this are custom columns
"""
import argparse
import sys
import numpy as np
import h5py
from datetime import datetime
import io

class TextParser:
    def __init__(self, input_file, output_file=None, ymd=True):
        """
        Constructor / initializer.
        :params: Dictionary of parameters for the XGBoost model
        """
        # Public instance variable
        self.input_file = input_file
        self.output_file = output_file
        self.ymd = ymd # should be a string

    def read_header(self,line):
        """
        Parse the header line of the form:
        Longitude = Xdeg Latitude = Ydeg
        Returns longitude (float degrees), latitude (float degrees).
        """
        parts = line.strip().split()
        try:
            lon_idx = parts.index('Longitude') + 2
            lat_idx = parts.index('Latitude') + 2
            lon = float(parts[lon_idx].rstrip('deg'))
            lat = float(parts[lat_idx].rstrip('deg'))
            return lon, lat
        except (ValueError, IndexError):
            raise ValueError(f"Header line not in expected format: {line}")


    def parseFile(self):
        # Read all lines
        if self.input_file is not None:
            # Wrap the binary buffer as text
            text_buf = io.TextIOWrapper(self.input_file, encoding='utf-8')
            header_line   = text_buf.readline()
            colnames_line = text_buf.readline()
            data_lines    = text_buf.readlines()


        # Parse and store coordinates
        longitude, latitude = self.read_header(header_line)

        # Parse column names
        colnames = colnames_line.strip().split()
        if len(colnames) < 6 or colnames[0] != 'Date' or colnames[1] != 'Time':
            raise ValueError(
                f"Column header line must start with 'Date Time ...': {colnames_line}")
        data_colnames = colnames[2:]

        # Final output column headers: year, month, day, hour, minute, second + data_colnames
        column_headers = ['year', 'month', 'day', 'hour', 'minute', 'second'] + data_colnames

        records = []  # list of lists of floats

        for lineno, line in enumerate(data_lines, start=3):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < len(colnames):
                print(f"Warning: line {lineno} has fewer columns than header, skipping.",
                    file=sys.stderr)
                continue

            # Date/Time parsing
            date_str, time_str = parts[0], parts[1]
            if self.ymd:
                try:
                    y, m, d = map(int, date_str.split('-'))
                    hh, mm, ss = map(int, time_str.split(':'))
                except Exception:
                    raise ValueError(f"Invalid YMD/time at line {lineno}: {date_str} {time_str}")
            else:
                try:
                    ts = float(date_str)
                    dt = datetime.utcfromtimestamp(ts)
                    y, m, d = dt.year, dt.month, dt.day
                    hh, mm, ss = dt.hour, dt.minute, dt.second
                except Exception:
                    raise ValueError(f"Invalid UNIX timestamp at line {lineno}: {date_str}")

            row_vals = [float(y), float(m), float(d), float(hh), float(mm), float(ss)]

            # Parse all other columns generically
            for idx, val in enumerate(parts[2:len(colnames)]):
                # Boolean check
                if val.lower() == 'true':
                    fval = 1.0
                elif val.lower() == 'false':
                    fval = 0.0
                else:
                    try:
                        fval = float(val)
                    except ValueError:
                        raise ValueError(f"Non-numeric value at line {lineno}, col {idx+3}: {val}")
                row_vals.append(fval)

            records.append(row_vals)

        # Convert to NumPy array
        data_array = np.array(records, dtype=float)

        # Save to HDF5
        if self.output_file is not None:
            with h5py.File(self.output_file, 'w') as h5f:
                h5f.create_dataset('observations', data=data_array)
                # store lat/lon as attributes and datasets
                h5f.attrs['longitude_deg'] = longitude
                h5f.attrs['latitude_deg'] = latitude
                h5f.create_dataset('longitude_deg', data=longitude)
                h5f.create_dataset('latitude_deg', data=latitude)
                # store column headers as a string dataset
                dt = h5py.string_dtype(encoding='utf-8')
                h5f.create_dataset('column_headers', data=np.array(column_headers, dtype=dt))

        # Print summary
        # Returns array of parsed data, lon/lat and column header names
        return data_array, longitude, latitude, column_headers