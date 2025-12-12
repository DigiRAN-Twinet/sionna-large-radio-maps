#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import numpy as np
import argparse

from common import add_project_root_to_path

add_project_root_to_path()
from sionna_lrm import RESULTS_DIR
from sionna_lrm.base_stations import load_transmitters
from sionna_lrm.tiling import create_tiling


def generate_tiling(
    output_file: str,
    bbox: tuple[float, float, float, float],
    min_size: float = 5,
    max_size: float = 100,
    shapefile: str | None = None,
):
    """
    Break a region into tiles based on base station density.

    Args:
        output_file: Path to output file.
        bbox: Bounding box as (south, west, north, east) in degrees.
        shapefile: Optional path to a shapefile to prune tiles.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(
            f"Invalid bounding box coordinates: {bbox}. Coordinates are expected to be in (min_lat, min_lon, max_lat, max_lon) format. If the region of interest crosses the antimeridian, please split it into two separate bounding boxes."
        )
    if min_lat < -90 or max_lat > 90 or min_lon < -180 or max_lon > 180:
        raise ValueError(
            f"Invalid bounding box coordinates: {bbox}. Latitude must be between -90 and 90 degrees, and longitude must be between -180 and 180 degrees."
        )

    lower_left = (min_lat, min_lon)
    upper_right = (max_lat, max_lon)
    target_stations_per_tile = 100

    _, transmitters_latlon = load_transmitters()

    tile_corners_latlons = create_tiling(
        lower_left,
        upper_right,
        min_tile_side_m=min_size * 1e3,
        max_tile_side_m=max_size * 1e3,
        base_stations_latlon=transmitters_latlon,
        target_stations_per_tile=target_stations_per_tile,
        restrict_to_shapefile=shapefile,
    )

    # Write out all of the tiles to a file
    output_fname = os.path.join(RESULTS_DIR, output_file)
    np.savez(output_fname, corners=tile_corners_latlons)
    print(f"[+] Saved {tile_corners_latlons.shape[0]} tile corners to: {output_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a tiling hierarchy for the US based on base station density.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_file", type=str, help="Output file for the tile corners."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        default=[24, -127, 50, -65],
        help="Bounding box of the region, in degrees. If not provided, defaults to the US mainland.",
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        help="Optional shapefile to prune tiles.",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=5,
        help="Minimum tile side length allowed, in km.",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=100,
        help="Maximum tile side length allowed, in km.",
    )

    args = parser.parse_args()
    generate_tiling(**vars(args))
