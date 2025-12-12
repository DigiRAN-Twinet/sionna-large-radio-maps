#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import pandas as pd
import numpy as np
import argparse

from common import add_project_root_to_path

add_project_root_to_path()
from sionna_lrm import SLRM_DATA_DIR
from sionna_lrm.constants import DEFAULT_TRANSMITTERS_FNAME

SELECTED_COUNTRY = "United States of America"
SELECTED_CARRIER = "Verizon Wireless"


def process_data(input_path: str, output_path: str):
    """
    Preprocess OpenCellID transmitter data CSV to:
        - keep only relevant columns
        - filter for 4G/5G in the US
        - add a column for elevation
    """

    df = pd.read_csv(input_path, usecols=["Country", "radio", "Network", "LAT", "LON"])
    df.insert(0, "building", False)
    df.insert(0, "elevation", np.nan)

    # Keep 4G and 5G only, in the US, and for Verizon
    df = df[
        (df["Country"] == SELECTED_COUNTRY)
        & ((df["radio"] == "LTE") | (df["radio"] == "NR"))
        & (df["Network"] == SELECTED_CARRIER)
    ]
    df = df.drop(columns=["Country", "radio"])
    df = df.rename(columns={"LAT": "lat", "LON": "lon"})

    # Remove duplicate antennas (same lat/lon)
    latlon = df[["lat", "lon"]].to_numpy()
    _, idx = np.unique(latlon, axis=0, return_index=True)
    df = df.iloc[idx]

    # Write out the final dataframe
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OpenCellID transmitter data CSV to keep only relevant columns and filter for 4G/5G in the US."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(
            SLRM_DATA_DIR, "remote", "transmitters", "opencellid", "north_america.csv"
        ),
        help="Path to the raw OpenCellID CSV file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TRANSMITTERS_FNAME,
        help="Path to save the processed CSV file.",
    )

    args = parser.parse_args()
    process_data(args.input, args.output)
