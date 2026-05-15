#!/usr/bin/env python3
"""
read_h5_and_print.py

Script to read an HDF5 file created by parse_to_h5.py, extract the 'observations'
dataset, metadata (latitude/longitude), column headers, and print them for verification.

Supports all extra columns and displays boolean-to-float conversion results.

Usage:
    python read_h5_and_print.py input.h5
"""
import argparse
import h5py
import numpy as np
import TextParser

# These should be changed based on parameters of the desired input file
input_file = 'unix_example.txt'
output_file = 'unix_output.h5'
ymd = False

parser = TextParser.TextParser(input_file, output_file, ymd)
storage = parser.parseFile()

print(storage)
