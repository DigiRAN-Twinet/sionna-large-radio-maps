#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas
from geopandas import GeoDataFrame
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pandas import IndexSlice
from scipy.spatial import KDTree
from shapely.geometry import box as shapely_box
from pyproj import Transformer

from .constants import (
    DEFAULT_LATLON_CRS,
    DEFAULT_TX_SEARCH_RADIUS_FACTOR,
    EARTH_RADIUS_M,
)


def haversine_distance(
    latlon1: tuple[float, float] | np.ndarray, latlon2: tuple[float, float] | np.ndarray
) -> float:
    """
    Calculate the great circle distance between two points on Earth's surface.

    Uses the Haversine formula to compute the distance between two latitude/longitude pairs.

    Returns:
        Distance in meters
    """
    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)

    # Convert degrees to radians
    lat1_rad = np.radians(latlon1[..., 0])
    lon1_rad = np.radians(latlon1[..., 1])
    lat2_rad = np.radians(latlon2[..., 0])
    lon2_rad = np.radians(latlon2[..., 1])

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def create_tiling(
    lower_left_latlon: tuple[float, float],
    upper_right_latlon: tuple[float, float],
    min_tile_side_m: float,
    max_tile_side_m: float,
    target_stations_per_tile: int = 100,
    base_stations_latlon: np.ndarray | None = None,
    search_radius_factor: float = DEFAULT_TX_SEARCH_RADIUS_FACTOR,
    restrict_to_shapefile: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Create a tiling of the given Earth surface region using square tiles.

    Similar to a quadtree building process, the tiles are subdivided recursively
    until the target number of base stations per tile is reached.

    Args:
        lower_left_latlon: latitude and longitude of the lower left corner of the bounding box,
                           in degrees.
        upper_right_latlon: latitude and longitude of the upper right corner of the bounding box,
                            in degrees.
        min_tile_side_m: minimum side length of a tile, in meters.
        max_tile_side_m: maximum side length of a tile, in meters.
        target_stations_per_tile: target number of base stations per tile.
        base_stations_latlon: list of base station coordinates, in degrees.
                              Used to determine the size of the tiles.
        restrict_to_shapefile: an optional path to a closed shapefile.
                               If given, only tiles that overlap with this shape will be kept.
        verbose: used to enable status logs during processing.

    Returns:
         A Numpy array of shape (n_tiles, 2, 2) containing for each tile:
            For each tile:
                For each corner (lower left, upper right):
                    Latitude, Longitude in degrees.
    """
    # Distance between corners of the bounding box
    # First is north-south, second and third are east-west along the south and north edges.
    distances_latlon_m = haversine_distance(
        np.array(
            [
                lower_left_latlon,
                lower_left_latlon,
                upper_right_latlon,
            ]
        ),
        np.array(
            [
                [upper_right_latlon[0], lower_left_latlon[1]],
                [lower_left_latlon[0], upper_right_latlon[1]],
                [upper_right_latlon[0], lower_left_latlon[1]],
            ]
        ),
    )
    assert distances_latlon_m.shape == (3,)

    d_lat_m = distances_latlon_m[0]
    d_lon_m = np.max(distances_latlon_m[1:])
    # Compute the number of full tiles in each dimension.
    n_tiles_x = 1 + int(np.ceil(d_lon_m / max_tile_side_m))
    n_tiles_y = 1 + int(np.ceil(d_lat_m / max_tile_side_m))

    # 1. Create a regular grid of tiles.
    # Compute latitudes and longitudes of each tile's corners.
    # Note: this most likely does not work for "wrapparound" cases.
    tile_lats, tile_lons = np.meshgrid(
        np.linspace(lower_left_latlon[0], upper_right_latlon[0], n_tiles_y),
        np.linspace(lower_left_latlon[1], upper_right_latlon[1], n_tiles_x),
    )
    tile_latlons = np.stack([tile_lats, tile_lons], axis=-1)
    assert tile_latlons.shape == (n_tiles_x, n_tiles_y, 2)

    # Convert to a layout where two corners are stored explicitly for each tile.
    # This is more redundant than the meshgrid-style layout, but will allow us
    # to remove entries, etc.,
    # New shape: (n_tiles_x, n_tiles_y, 2, 2), where the last two dimensions are:
    #     - corner index (lower-left, upper-right)
    #     - lat/lon
    tile_corners_latlon = np.stack(
        [
            tile_latlons[:-1, :-1, :],
            tile_latlons[1:, 1:, :],
        ],
        axis=-2,
    )
    # Since we will now start eliminating some tiles, switch to a linear list.
    tile_corners_latlon = tile_corners_latlon.reshape(-1, 2, 2)

    # 2. Subdivide the tiles until we reach the target number of base stations per tile.
    if base_stations_latlon is not None:
        tile_corners_latlon = refine_tiling_for_positions(
            tile_corners_latlon,
            base_stations_latlon,
            min_tile_side_m,
            target_stations_per_tile,
            search_radius_factor,
            verbose=verbose,
        )

    # 3. Eliminate tiles that are fully not on land.
    if restrict_to_shapefile is not None:
        preliminary_box = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely_box(
                        lower_left_latlon[1],
                        lower_left_latlon[0],
                        upper_right_latlon[1],
                        upper_right_latlon[0],
                    )
                ]
            },
            crs="EPSG:4326",
        )
        tile_corners_latlon, filtered_df, _ = restrict_to_overlap_with_shapefile(
            restrict_to_shapefile, tile_corners_latlon, preliminary_box=preliminary_box
        )
        if base_stations_latlon is not None:
            filtered_idx = filtered_df.index.to_numpy()

    return tile_corners_latlon


def refine_tiling_for_positions(
    tile_corners_latlons: np.ndarray,
    base_stations_latlon: np.ndarray,
    min_tile_side_m: float,
    target_stations_per_tile: int,
    search_radius_factor: float,
    max_subdivisions: int = 15,
    verbose: bool = False,
) -> np.ndarray:
    assert tile_corners_latlons.ndim == 3 and tile_corners_latlons.shape[1:] == (2, 2)

    # Projection to ECEF for distance computations
    wgs84_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

    tx_ecef = np.stack(
        wgs84_to_ecef.transform(
            base_stations_latlon[:, 1],
            base_stations_latlon[:, 0],
            np.zeros_like(base_stations_latlon[:, 0]),
        ),
        axis=-1,
    )

    tile_corners_ecef = np.stack(
        wgs84_to_ecef.transform(
            tile_corners_latlons[..., 1],
            tile_corners_latlons[..., 0],
            np.zeros_like(tile_corners_latlons[..., 0]),
        ),
        axis=-1,
    )

    tx_tree = KDTree(data=tx_ecef)
    min_tile_area_m2 = min_tile_side_m * min_tile_side_m

    for i in range(max_subdivisions):
        cell_centers_wgs84 = 0.5 * (
            tile_corners_latlons[:, 0] + tile_corners_latlons[:, 1]
        )
        cell_centers_ecef = np.stack(
            wgs84_to_ecef.transform(
                cell_centers_wgs84[:, 1],
                cell_centers_wgs84[:, 0],
                np.zeros_like(cell_centers_wgs84[:, 0]),
            ),
            axis=-1,
        )

        current_widths_m = haversine_distance(
            tile_corners_latlons[..., 0, :],
            np.stack(
                [tile_corners_latlons[..., 0, 0], tile_corners_latlons[..., 1, 1]],
                axis=-1,
            ),
        )
        current_heights_m = haversine_distance(
            tile_corners_latlons[..., 0, :],
            np.stack(
                [tile_corners_latlons[..., 1, 0], tile_corners_latlons[..., 0, 1]],
                axis=-1,
            ),
        )
        current_areas_m2 = current_widths_m * current_heights_m

        search_radii = (
            search_radius_factor
            * 0.5
            * np.linalg.norm(
                tile_corners_ecef[..., 1, :] - tile_corners_ecef[..., 0, :], axis=-1
            )
        )

        # Count transmitters within the tile
        n_neighbors = tx_tree.query_ball_point(
            cell_centers_ecef, r=search_radii, return_length=True
        )

        # Subdivision criteria:
        # - Subdividing shouldn't take us below the min tile area threshold
        # - Don't need to subdivide if we have already reached our
        #   target number of transmitters per tile.
        subdivide = (current_areas_m2 > 4 * min_tile_area_m2) & (
            n_neighbors > target_stations_per_tile
        )
        n_subdivisions = subdivide.sum()
        if n_subdivisions == 0:
            break
        if verbose:
            print(f"[{i:02d}] Subdividing {n_subdivisions} tiles")

        # Divide each subdivided tile into 4 subtiles.
        # Note: tiles will no longer be nicely ordered starting from the first subdivision.
        untouched = tile_corners_latlons[~subdivide, ...]
        to_subdivide = tile_corners_latlons[subdivide, ...]
        relevant_centers = cell_centers_wgs84[subdivide, ...]

        bottom_left = np.stack([to_subdivide[:, 0, :], relevant_centers], axis=1)
        bottom_right = np.stack(
            [
                [relevant_centers[:, 0], to_subdivide[:, 0, 1]],
                [to_subdivide[:, 1, 0], relevant_centers[:, 1]],
            ],
            axis=1,
        ).T
        top_left = np.stack(
            [
                [to_subdivide[:, 0, 0], relevant_centers[:, 1]],
                [relevant_centers[:, 0], to_subdivide[:, 1, 1]],
            ],
            axis=1,
        ).T
        top_right = np.stack([relevant_centers, to_subdivide[:, 1, :]], axis=1)

        tile_corners_latlons = np.concat(
            [
                untouched,
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            ],
            axis=0,
        )
        assert tile_corners_latlons.shape == (
            untouched.shape[0] + 4 * n_subdivisions,
            *untouched.shape[1:],
        )

        tile_corners_ecef = np.stack(
            wgs84_to_ecef.transform(
                tile_corners_latlons[..., 1],
                tile_corners_latlons[..., 0],
                np.zeros_like(tile_corners_latlons[..., 0]),
            ),
            axis=-1,
        )

    if verbose:
        print(f"[+] After subdivision, there are {tile_corners_ecef.shape[0]} tiles")
    return tile_corners_latlons


def tile_corners_latlon_to_geopandas(
    box_corners_latlons: np.ndarray, crs: str = DEFAULT_LATLON_CRS
) -> GeoDataFrame:
    return GeoDataFrame(
        {
            "geometry": [
                shapely_box(
                    minx=tile_corners_latlon[0, 1],
                    miny=tile_corners_latlon[0, 0],
                    maxx=tile_corners_latlon[1, 1],
                    maxy=tile_corners_latlon[1, 0],
                )
                for tile_corners_latlon in box_corners_latlons[:, ...]
            ]
        },
        crs=crs,
    )


def geodataframe_to_tile_corners_latlon(
    df: GeoDataFrame, latlon_crs: str = DEFAULT_LATLON_CRS
) -> np.ndarray:
    """
    Returns lower-left and upper-right corners of each tile in the given GeoDataFrame.
    """
    result_df = df.to_crs(latlon_crs)
    result_df = result_df.geometry.exterior.get_coordinates(index_parts=True)

    corners_latlon = [result_df.loc[IndexSlice[:, i], ["y", "x"]] for i in range(4)]
    min_latlon = np.min(corners_latlon, axis=0)
    max_latlon = np.max(corners_latlon, axis=0)

    corners_np = np.stack(
        [min_latlon, max_latlon],
        axis=1,
    )

    assert corners_np.ndim == 3 and corners_np.shape[1:] == (2, 2), corners_np.shape
    assert np.all(corners_np[:, 0, :] <= corners_np[:, 1, :])

    return corners_np


def restrict_to_overlap_with_bbox(
    df: GeoDataFrame,
    lower_left_latlon: tuple[float, float],
    upper_right_latlon: tuple[float, float],
) -> tuple[GeoDataFrame, GeoDataFrame]:
    preliminary_box = geopandas.GeoDataFrame(
        {
            "geometry": [
                shapely_box(
                    lower_left_latlon[1],
                    lower_left_latlon[0],
                    upper_right_latlon[1],
                    upper_right_latlon[0],
                )
            ]
        },
        crs="EPSG:4326",
    )
    preliminary_box = preliminary_box.to_crs(df.crs)
    df = df.copy()
    df["index"] = df.index
    return df.overlay(preliminary_box, how="intersection"), preliminary_box


def restrict_to_overlap_with_shapefile(
    shapefile: str,
    box_corners_latlons: np.ndarray,
    preliminary_box: GeoDataFrame | None = None,
    show: bool = False,
) -> tuple[np.ndarray, GeoDataFrame, GeoDataFrame]:
    shapefile_df = geopandas.read_file(shapefile)

    # If given, apply a preliminary bounding box
    if preliminary_box is not None:
        preliminary_box = preliminary_box.to_crs(shapefile_df.crs)
        shapefile_df = shapefile_df.overlay(preliminary_box, how="intersection")

    tiles_df = tile_corners_latlon_to_geopandas(box_corners_latlons)
    tiles_df = tiles_df.to_crs(shapefile_df.crs)

    # Note: `intersects` checks intersections between matching rows, so we just
    # replicate the US shape once per tile.
    does_intersect = tiles_df.intersects(
        shapefile_df.loc[[0] * tiles_df.shape[0]], align=False
    )
    filtered_df = tiles_df[does_intersect]

    if show:
        _, ax = plt.subplots(1, 1, figsize=(9, 13))
        shapefile_df.plot(ax=ax, cmap="Accent")
        filtered_df.plot(ax=ax, alpha=0.5, edgecolor="black")

    # Convert back to a plain (lat, lon) array for tile corners
    corners_np = geodataframe_to_tile_corners_latlon(filtered_df)

    return corners_np, filtered_df, shapefile_df


def visualize_tiling(
    tile_corners_latlon: np.ndarray,
    as_rectangles: bool = False,
    rect_kwargs: dict[str, Any] | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axis | None = None,
    draw_basemap: bool = True,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    if (fig is not None) or (ax is not None):
        assert (fig is not None) and ax is not None
    else:
        fig, ax = plt.subplots(figsize=(9, 10))

    mapping = Basemap(
        projection="merc",
        llcrnrlat=tile_corners_latlon[..., 0].min(),
        llcrnrlon=tile_corners_latlon[..., 1].min(),
        urcrnrlat=tile_corners_latlon[..., 0].max(),
        urcrnrlon=tile_corners_latlon[..., 1].max(),
        ax=ax,
        resolution="i",
    )

    tiles_xs, tiles_ys = mapping(
        tile_corners_latlon[..., 1], tile_corners_latlon[..., 0]
    )
    tile_corners_xys = np.stack([tiles_xs, tiles_ys], axis=-1)
    assert tile_corners_latlon.ndim == 3
    assert tile_corners_latlon.shape[-2:] == (2, 2)
    assert tile_corners_xys.shape == tile_corners_latlon.shape

    if draw_basemap:
        mapping.drawcoastlines(linewidth=0.25)
        mapping.drawcountries(linewidth=0.25)
        mapping.fillcontinents(
            color=(0.388, 0.529, 0.125), lake_color=(0.078, 0.325, 0.565)
        )
        mapping.drawmapboundary(fill_color=(0.078, 0.325, 0.565))

    if as_rectangles:
        # Draw rectangles for each tile
        rect_kwargs_ = {
            "linewidth": 0.5,
            "edgecolor": "red",
            "facecolor": "none",
            "alpha": 0.7,
        }
        if rect_kwargs is not None:
            rect_kwargs_.update(rect_kwargs)

        for rect_i in range(tile_corners_xys.shape[0]):
            rect_start = tile_corners_xys[rect_i, 0, :]
            width = tile_corners_xys[rect_i, 1, 0] - rect_start[0]
            height = tile_corners_xys[rect_i, 1, 1] - rect_start[1]

            rect_kwargs_i = rect_kwargs_.copy()
            for k, v in rect_kwargs_i.items():
                if isinstance(v, list):
                    rect_kwargs_i[k] = v[rect_i]

            rect = Rectangle(
                rect_start,
                width,
                height,
                **rect_kwargs_i,
            )
            ax.add_patch(rect)
    else:
        tiles_x0 = tile_corners_xys[..., 0, 1].ravel()
        tiles_y0 = tile_corners_xys[..., 0, 0].ravel()
        tiles_x1 = tile_corners_xys[..., 1, 1].ravel()
        tiles_y1 = tile_corners_xys[..., 1, 0].ravel()
        ax.plot(tiles_x0, tiles_y0, tiles_x1, tiles_y1, color="black", alpha=0.5)

    return fig, ax, tile_corners_xys
