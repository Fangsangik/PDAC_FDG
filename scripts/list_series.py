"""Print the series inventory of one DICOM study directory so a human can pick the CT/PET series.
usage: python scripts/list_series.py <patient_dicom_dir> [--csv out.csv]"""
import argparse

import pandas as pd

from fdgmip.io import list_series, series_table

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--csv")
    a = ap.parse_args()
    tab = series_table(list_series(a.directory))
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(tab.to_string(index=False))
    if a.csv:
        tab.to_csv(a.csv, index=False)
