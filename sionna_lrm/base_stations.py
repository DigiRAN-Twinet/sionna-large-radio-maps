#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import KDTree
from tqdm import tqdm

from .constants import (
    SLRM_DATA_DIR,
    DEFAULT_LATLON_ELLIPSOID,
    DEFAULT_TRANSMITTERS_FNAME,
)


def search_query_for_region(
    lower_left_latlon: tuple[float, float],
    upper_right_latlon: tuple[float, float],
    search_extra_m: float,
    search_radius_factor: float,
    return_updated_corners: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Returns:
        np.ndarray: the center of the search region.
        float: the search radius.
        np.ndarray: (optional) the updated lower left corner of the search region.
        np.ndarray: (optional) the updated upper right corner of the search region.
    """
    lower_left_latlon = np.array(lower_left_latlon)
    upper_right_latlon = np.array(upper_right_latlon)

    # Since the area of effect of a transmitter does not depend on the current
    # tile's size, we include a fixed-size extra distance beyond the tile boundary.
    if search_extra_m > 0:
        # We will extend the tile corners by the extra distance on each side.
        geod = Geod(ellps=DEFAULT_LATLON_ELLIPSOID)

        n = 1 if lower_left_latlon.ndim == 1 else lower_left_latlon.shape[0]
        search_extra_m = np.full(n, search_extra_m)

        # Note: azimuth (absolute bearing in degrees) determines the direction of travel
        lower_left_latlon[..., 1], lower_left_latlon[..., 0], _ = geod.fwd(
            lower_left_latlon[..., 1],
            lower_left_latlon[..., 0],
            az=np.full(n, 225.0),
            dist=search_extra_m,
        )
        upper_right_latlon[..., 1], upper_right_latlon[..., 0], _ = geod.fwd(
            upper_right_latlon[..., 1],
            upper_right_latlon[..., 0],
            az=np.full(n, 45.0),
            dist=search_extra_m,
        )

    # Compute search radius
    d_lat = upper_right_latlon[..., 0] - lower_left_latlon[..., 0]
    d_lon = upper_right_latlon[..., 1] - lower_left_latlon[..., 1]
    r = search_radius_factor * 0.5 * np.linalg.norm((d_lat, d_lon))

    center = np.stack(
        [
            0.5 * (lower_left_latlon[..., 0] + upper_right_latlon[..., 0]),
            0.5 * (lower_left_latlon[..., 1] + upper_right_latlon[..., 1]),
        ],
        axis=-1,
    )
    assert center.shape[-1] == 2

    if return_updated_corners:
        return center, r, lower_left_latlon, upper_right_latlon

    return center, r


class BaseStationDB:
    """
    A class that manages a database of base stations.
    """

    def __init__(self, tx: pd.DataFrame):
        """
        Initialize the database given a Pandas DataFrame containing the base station data.
        The DataFrame must contain the following columns:
        - 'lat': latitude in degrees
        - 'lon': longitude in degrees
        - 'elevation': elevation in meters (can be NaN)
        - 'building': boolean indicating if the base station is over a building
        """
        self.tx_df = tx
        self.tx_count = len(tx)
        # Build a KDTree for fast spatial queries
        self.tx_tree = KDTree(self.tx_df[["lat", "lon"]].to_numpy())

    def __len__(self):
        return self.tx_count

    @classmethod
    def from_file(cls, fname: str = DEFAULT_TRANSMITTERS_FNAME) -> BaseStationDB:
        """
        Initialize the database by loading the base station data from a CSV file.
        The CSV file must contain the following columns:
        - 'lat': latitude in degrees
        - 'lon': longitude in degrees
        - 'elevation': elevation in meters (can be NaN)
        - 'building': boolean indicating if the base station is over a building
        """
        tx = pd.read_csv(fname)
        if (
            "lat" not in tx.columns
            or "lon" not in tx.columns
            or "elevation" not in tx.columns
        ):
            raise ValueError(
                "CSV file must contain 'lat', 'lon', and 'elevation' columns."
            )
        return cls(tx)

    @classmethod
    def from_json(
        cls, fname: str | None = None, show_progress: bool = True
    ) -> BaseStationDB:
        if fname is None:
            fname = os.path.join(
                SLRM_DATA_DIR, "remote", "transmitters", "tower-maps", "data.json"
            )

        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        tx_info = []
        for feature in tqdm(
            data["features"], desc="Loading base stations", disable=not show_progress
        ):
            lat, lon = feature["geometry"]["y"], feature["geometry"]["x"]
            elevation = feature["attributes"].get("ELEVATION", np.nan)
            tx_info.append([lat, lon, elevation, True])  # Assume building=True

        tx_df = pd.DataFrame(
            np.array(tx_info), columns=["lat", "lon", "elevation", "building"]
        )
        return cls(tx_df)

    def to_file(self, fname: str) -> None:
        """
        Save the base station data to a CSV file.

        Parameters:
            fname: path to the output CSV file
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.tx_df.to_csv(fname, index=False)

    def elevation(self) -> np.ndarray:
        return self.tx_df["elevation"].to_numpy()

    def latitude(self) -> np.ndarray:
        return self.tx_df["lat"].to_numpy()

    def longitude(self) -> np.ndarray:
        return self.tx_df["lon"].to_numpy()

    def latlon(self) -> np.ndarray:
        return self.tx_df[["lat", "lon"]].to_numpy()

    def is_over_building(self) -> np.ndarray:
        return self.tx_df["building"].to_numpy()

    def index_at(self, idx: int | np.ndarray) -> int | np.ndarray:
        """
        Get the global index of the base station at the given local index.
        Useful when the database has been filtered down to a subset.
        """
        result = self.tx_df.index[idx]
        if isinstance(idx, int):
            return result
        return result.to_numpy()

    def set_over_building(self, idx: np.array, over_building: np.array) -> None:
        """
        Set the elevation of the base stations at the given indices.

        Parameters:
            idx: indices of the base stations to update
            elevation: new elevation values in meters
        """
        self.tx_df.loc[idx, "building"] = over_building

    def set_elevation(self, idx: np.array, elevation: np.array) -> None:
        """
        Set the elevation of the base stations at the given indices.

        Parameters:
            idx: indices of the base stations to update
            elevation: new elevation values in meters
        """
        self.tx_df.loc[idx, "elevation"] = elevation

    def get_region(
        self,
        lower_left_latlon: tuple[float, float],
        upper_right_latlon: tuple[float, float],
        search_extra_m: float = 0.0,
        search_radius_factor: float = 1.0,
        restrict_to_latlon_bbox: bool = False,
        return_idx: bool = False,
    ) -> BaseStationDB:
        """
        Filters the transmitter data to only include those within the given region.

        Parameters:
            lower_left_latlon: (lat, lon) of the lower-left corner of the region
            upper_right_latlon: (lat, lon) of the upper-right corner of the region
            search_extra_m: extra distance in meters to search for transmitters beyond the tile boundary
            search_radius_factor: factor by which to multiply the search radius

        Returns:
            BaseStationDB: a new BaseStationDB instance containing only the transmitters within the region

        """
        # Get the indices of transmitters that are close enough to the zone of interest
        center, r = search_query_for_region(
            lower_left_latlon, upper_right_latlon, search_extra_m, search_radius_factor
        )

        # Perform the search
        tx_idx = np.array(self.tx_tree.query_ball_point(center, r))

        tx_latlon = self.tx_df.iloc[tx_idx][["lat", "lon"]].to_numpy()
        if restrict_to_latlon_bbox:
            valid = (
                (tx_latlon[:, 0] >= lower_left_latlon[0])
                & (tx_latlon[:, 0] <= upper_right_latlon[0])
                & (tx_latlon[:, 1] >= lower_left_latlon[1])
                & (tx_latlon[:, 1] <= upper_right_latlon[1])
            )
            tx_idx = tx_idx[valid]

        region_db = BaseStationDB(self.tx_df.iloc[tx_idx])
        if return_idx:
            return region_db, tx_idx
        return region_db


class BaseStationType(Enum):
    BUILDING = "Building"
    ROOFTOP = "Rooftop"
    LAND = "Land"
    TOWER_MONOPOLE = "Tower (Monopole)"
    TOWER_SELF_SUPPORTING = "Tower (Self-Supporting)"
    TOWER_LATICE = "Tower (Lattice)"
    TOWER_GUYED = "Tower (Guyed)"
    TOWER_OTHER = "Tower (Other)"
    UTILITY_STRUCTURE = "Utility Structure"
    BILLBOARD = "Billboard"
    TBD = "TBD"
    N_A = "N/A"
    OTHER = "Other"


@dataclass
class BaseStation:
    object_id: int
    #: Height above ground level in meters (?). It may not be filled-in correctly.
    above_ground_level: float
    #: Actual elevation computed from raycasting
    elevation: float | None
    type: BaseStationType
    lat: float
    lon: float


def load_transmitters(
    fname: str | None = None, show_progress: bool = True
) -> tuple[list[BaseStation], np.ndarray]:
    """
    The returned latitude and longitude are in EPSG:4326.
    """
    if fname is None:
        fname = os.path.join(
            SLRM_DATA_DIR, "remote", "transmitters", "tower-maps", "data.json"
        )

    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_stations = []
    coordinates = []
    for feature in tqdm(
        data["features"], desc="Loading base stations", disable=not show_progress
    ):
        latlon = [feature["geometry"]["y"], feature["geometry"]["x"]]
        agl = feature["attributes"]["AGL"]
        if agl is None:
            agl = 0
        agl = float(agl)
        elevation = feature["attributes"].get("ELEVATION", None)

        tp = feature["attributes"]["TYPE_"]
        tp = BaseStationType(tp) if (tp is not None) else BaseStationType.N_A

        base_stations.append(
            BaseStation(
                object_id=feature["attributes"]["OBJECTID"],
                above_ground_level=agl,
                type=tp,
                lat=latlon[0],
                lon=latlon[1],
                elevation=elevation,
            )
        )
        coordinates.append(latlon)

    coordinates = np.array(coordinates)
    return base_stations, coordinates
