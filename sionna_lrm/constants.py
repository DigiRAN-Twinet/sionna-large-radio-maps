#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import os

from matplotlib import colormaps as cm

from . import SLRM_DATA_DIR


# Earth radius in meters.
EARTH_RADIUS_M: float = 6371000.0
# Default CRS used to represent latitude and longitude. Uses the WGS84 reference ellipsoid.
DEFAULT_LATLON_CRS: str = "EPSG:4326"
DEFAULT_LATLON_ELLIPSOID: str = "WGS84"


# Default name (ID) for the measurement surface in the tile scenes.
DEFAULT_MEASUREMENT_MESH_NAME = "ground"

# Search radius factor for the transmitters to include with a tile.
DEFAULT_TX_SEARCH_RADIUS_FACTOR = 1.5
# Extra distance in meters to search for transmitters beyond the tile boundary.
DEFAULT_TX_SEARCH_DISTANCE_M = 2000.0

# Vertical offsets to place the transmitters, in meters.
DEFAULT_Z_OFFSET_BUILDING = 2.0
DEFAULT_Z_OFFSET_NON_BUILDING = 25.0

# Path to the transmitters file.
DEFAULT_TRANSMITTERS_FNAME = os.path.join(
    SLRM_DATA_DIR,
    "remote",
    "transmitters",
    "data.csv",  # UPDATE THIS
)

# Results visualization
DEFAULT_RM_CMAP = cm.get("viridis").copy()
DEFAULT_RM_DB_VMIN = -120.0
DEFAULT_RM_DB_VMAX = -45.0


# Reasonable assumption for the size of a scene once loaded (MB).
# Used to estimate memory usage when determining the number of passes required.
# Note that we may have to store multiple copies of the scene / measurement surface in memory.
ASSUMED_SCENE_SIZE_MIB = os.environ.get("ASSUMED_SCENE_SIZE_MIB", 3 * 1024)
# Minimum number of samples (paths) we would like to trace per transmitter.
MIN_SAMPLES_PER_TX = 20000000

DEFAULT_ANTENNA_ARRAY_PARAMS = {
    "num_rows": 2,
    "num_cols": 16,
    "vertical_spacing": 0.5,
    "horizontal_spacing": 0.5,
    "pattern": "tr38901",
    "polarization": "V",
}

# Default minimum and maxiumum tile size for tiling (in meters).
DEFAULT_MIN_CELL_SIZE = 5
DEFAULT_MAX_CELL_SIZE = 100
