#!/usr/bin/env python
"""
check_csv.py

Opens any CSV and prints requested columns to the terminal.

Usage:
    python check_csv.py <csv_path> [col1 col2 ...]

Examples:
    # Print the ID, ra and dec columns
    python check_csv.py /ceph/cephfs/apatrick/P2/jwst_catalogs/duplicate_sources.csv ID ra dec

    # No columns given -> list all column names, then print the whole table
    python check_csv.py /ceph/cephfs/apatrick/P2/jwst_catalogs/duplicate_sources.csv
"""

import os
import sys
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Open a CSV and print requested columns to the terminal."
    )
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument(
        "columns",
        nargs="*",
        help="Column names to print. If none given, all columns are shown.",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv_path)

    print("Opening CSV:")
    print("  " + csv_path)
    print("")

    if not os.path.exists(csv_path):
        print("ERROR: file not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    print("Rows: " + str(len(df)) + "   Columns: " + str(df.shape[1]))

    # Show everything, no truncation
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    if not args.columns:
        print("No columns requested, printing the full table:")
        print("")
        print(df.to_string(index=False))
        return

    # Check requested columns exist
    missing = [c for c in args.columns if c not in df.columns]
    if missing:
        print("WARNING: these requested columns are not in the CSV: " +
              ", ".join(missing))
        args.columns = [c for c in args.columns if c in df.columns]
        print("")

    if not args.columns:
        print("None of the requested columns exist. Nothing to print.")
        return

    print("Printing columns: " + ", ".join(args.columns))
    print("")
    print(df[args.columns].to_string(index=False))


if __name__ == "__main__":
    main()