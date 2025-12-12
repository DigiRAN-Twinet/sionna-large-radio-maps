# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# This file contains code adapted from geo2sigmap
# https://github.com/functions-lab/geo2sigmap

from functools import partial
import math
from multiprocessing import Pool
import os
from os.path import relpath, join, isfile, isdir, realpath
import shutil

import numpy as np
import pyproj
from pyproj import Transformer, CRS
from shapely.geometry import Polygon
from tqdm import tqdm

from sionna_lrm import LOCAL_SCENES_DIR, REMOTE_SCENES_DIR
from sionna_lrm.scene.heightmap import HeightMap


def bbox_to_mesh(
    bbox: tuple[float, float, float, float],
    center: tuple[float, float] | None = None,
    heightmap: HeightMap | None = None,
    to_utm: Transformer | None = None,
    target_crs: str = "UTM",
    resolution: float = 100.0,
    h_offset: float = 0.0,
    auto_heightmap: bool = True,
):
    """
    Create a ground mesh from the given bounding box and heightmap.

    Parameters
    ----------
    bbox : tuple of float
        Bounding box defined as (min_lat, min_lon, max_lat, max_lon), in degres.
    center : tuple of float
        Origin of the local frame in UTM. If None, use the bottom-left corner of the bbox.
    heightmap : HeightMap
        HeightMap object to query heights. If None, creates a flat mesh.
    to_utm : Transformer
        Transformer object to convert from WGS84 to UTM coordinates.
    target_crs : str
        Target system in which to create the mesh. Defaults to UTM.
    resolution : float, optional
        Grid resolution in meters (default: 100.0).
    h_offset : float, optional
        Height offset to add to the terrain (default: 0.0).
    """
    # bounding box size in UTM
    min_lat, min_lon, max_lat, max_lon = bbox
    if to_utm is None:
        projection_utm_epsg_code = get_utm_epsg_code_from_gps(min_lon, min_lat)
        to_utm = Transformer.from_crs(
            "EPSG:4326", projection_utm_epsg_code, always_xy=True
        )

    if target_crs == "UTM":
        transformer = to_utm
    elif target_crs == "WGS84":
        # Correct for EGM96 heights (EPSG:5773)
        transformer = Transformer.from_crs(
            "EPSG:4326+5773", "EPSG:4979", always_xy=True
        )
    elif target_crs == "ECEF":
        # Correct for EGM96 heights (EPSG:5773)
        transformer = Transformer.from_crs(
            "EPSG:4326+5773", "EPSG:4978", always_xy=True
        )
    else:
        raise ValueError(
            f"Unknown target_crs: {target_crs}. Must be one of [UTM, WGS84, ECEF]."
        )

    if center is None:
        if target_crs == "UTM":
            # Assume origin is the south-west corner of the region
            center = np.array(transformer.transform(min_lon, min_lat, 0.0))
        else:
            # Centering doesn't make a lot of sense in WGS84 or ECEF
            center = np.zeros(3)

    corners = np.array(
        [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat]]
    )
    x_utm, y_utm = to_utm.transform(corners[:, 0], corners[:, 1])
    res_x = max(10, int(math.ceil((x_utm.max() - x_utm.min()) / resolution)) + 1)
    res_y = max(10, int(math.ceil((y_utm.max() - y_utm.min()) / resolution)) + 1)
    # Auto-create heightmap if requested
    if heightmap is None and auto_heightmap:
        earth_circumference = 40075000  # meters
        lat_factor = np.cos(np.deg2rad(max_lat + min_lat) / 2)
        d_x = (x_utm.max() - x_utm.min()) / (res_x - 1)
        d_y = (y_utm.max() - y_utm.min()) / (res_y - 1)
        z_y = int(math.ceil(np.log2(earth_circumference / 512 / d_y)))
        z_x = int(math.ceil(np.log2(earth_circumference * lat_factor / 512 / d_x)))
        z = max(z_x, z_y)
        heightmap = HeightMap(utm_zone=to_utm.target_crs, bbox=bbox, z=z, parallel=True)

    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, res_x), np.linspace(0.0, 1.0, res_y), indexing="xy"
    )
    lon = min_lon + u.flatten() * (max_lon - min_lon)
    lat = min_lat + v.flatten() * (max_lat - min_lat)
    if heightmap is None:
        # Flat terrain at h_offset
        z = np.full_like(lon, h_offset)
    else:
        z = heightmap.height_from_wgs84(lon, lat) + h_offset
    x, y, z = transformer.transform(lon, lat, z)
    vertices = np.stack([x, y, z], axis=-1)
    vertices -= center
    idx = np.arange(res_x * res_y).reshape(res_y, res_x)
    i0 = idx[:-1, :-1].flatten()
    i1 = idx[:-1, 1:].flatten()
    i2 = idx[1:, :-1].flatten()
    i3 = idx[1:, 1:].flatten()
    faces = np.concatenate(
        [np.stack([i0, i1, i2], axis=-1), np.stack([i1, i3, i2], axis=-1)], axis=0
    )

    return vertices, faces


def get_utm_epsg_code_from_gps(lon: float, lat: float) -> CRS:
    """
    Determine the UTM coordinate reference system (CRS) appropriate for a given
    longitude/latitude using WGS84 as the datum.

    This function queries pyproj's database for the UTM zone that best fits
    the point of interest (defined by lon/lat).

    Parameters:
    ----------
    lon : float
        Longitude in decimal degrees.
    lat : float
        Latitude in decimal degrees.

    Returns:
    -------
    utm_crs : CRS
        A pyproj CRS object representing the best matching UTM projection
        (e.g., EPSG:32633).
    """

    # Query for possible UTM CRS definitions covering our point of interest
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    # Typically, the first element is the most relevant match
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def gps_to_utm_xy(lon: float, lat: float, utm_epsg):
    """
    Convert GPS coordinates (longitude, latitude) in WGS84 to UTM coordinates.

    Parameters:
    ----------
    lon : float
        Longitude in decimal degrees (WGS84).
    lat : float
        Latitude in decimal degrees (WGS84).
    utm_epsg : int
        The EPSG code for the desired UTM zone (e.g., 32633).

    Returns:
    -------
    (utm_x, utm_y, epsg_code) : (float, float, int)
        utm_x  : easting in the specified UTM zone
        utm_y  : northing in the specified UTM zone
        epsg_code : same as the input `utm_epsg`, returned for convenience
    """

    # Create a transformer from WGS84 (EPSG:4326) to the specified UTM zone
    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    # Transform (longitude, latitude) into (easting, northing) in the UTM zone
    utm_x, utm_y = transformer.transform(lon, lat)

    # Return the results, including the EPSG code for clarity
    return (utm_x, utm_y, utm_epsg)


def rect_from_point_and_size(
    lon: float, lat: float, position: str, width: float, height: float
) -> list[tuple[float, float]]:
    """
    Create a rectangular polygon (as a list of coordinates) given a GPS point and
    desired rectangle size in a UTM projection.

    :param lon: Longitude of the reference point (in EPSG:4326).
    :param lat: Latitude of the reference point (in EPSG:4326).
    :param position: One of the following strings:
                     ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
                     indicating how (lon, lat) is interpreted relative to the rectangle.
    :param width:  The width of the rectangle in UTM projection units (e.g., meters).
    :param height: The height of the rectangle in UTM projection units (e.g., meters).
    :return:       A list of (longitude, latitude) coordinates (EPSG:4326)
                   forming the rectangle boundary. The last point is repeated
                   to close the polygon.

    .. note::
       This function currently does NOT handle edge cases such as crossing
       the International Date Line or spanning multiple UTM zones. Those must
       be addressed separately.
    """

    # Get the UTM EPSG code based on the given longitude/latitude.
    utm_epsg = get_utm_epsg_code_from_gps(lon, lat)

    # Convert the reference GPS point to UTM (x, y).
    point_utm = gps_to_utm_xy(lon, lat, utm_epsg)

    # Prepare a transformer to go from UTM back to EPSG:4326.
    transformer = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

    if position == "top-left":
        min_lon_left = point_utm[0]
        max_lon_right = point_utm[0] + width
        max_lat_top = point_utm[1]
        min_lat_bottom = point_utm[1] - height

    elif position == "top-right":
        min_lon_left = point_utm[0] - width
        max_lon_right = point_utm[0]
        max_lat_top = point_utm[1]
        min_lat_bottom = point_utm[1] - height

    elif position == "bottom-right":
        min_lon_left = point_utm[0] - width
        max_lon_right = point_utm[0]
        max_lat_top = point_utm[1] + height
        min_lat_bottom = point_utm[1]

    elif position == "bottom-left":
        min_lon_left = point_utm[0]
        max_lon_right = point_utm[0] + width
        max_lat_top = point_utm[1] + height
        min_lat_bottom = point_utm[1]

    elif position == "center":
        min_lon_left = point_utm[0] - width / 2
        max_lon_right = point_utm[0] + width / 2
        max_lat_top = point_utm[1] + height / 2
        min_lat_bottom = point_utm[1] - height / 2

    else:
        raise ValueError(
            f"Unknown position: {position}. "
            "Must be one of [top-left, top-right, bottom-left, bottom-right, center]."
        )

    points_utm = [
        [min_lon_left, min_lat_bottom],
        [min_lon_left, max_lat_top],
        [max_lon_right, max_lat_top],
        [max_lon_right, min_lat_bottom],
        [min_lon_left, min_lat_bottom],
    ]

    points_gps = [transformer.transform(x, y) for x, y in points_utm]
    return points_gps


def unique_coords(input_coords):
    """
    Given a list of (x, y) coordinates, return a new list with duplicate
    coordinates removed, preserving the original order of first occurrences.

    Parameters
    ----------
    input_coords : list of (float, float)
        A list of 2D coordinate pairs.

    Returns
    -------
    list of (float, float)
        The same coordinates but with duplicates removed in order of appearance.
    """
    unique_coords_res = []
    seen_coords = set()
    for coord in input_coords:
        if coord not in seen_coords:
            unique_coords_res.append(coord)
            seen_coords.add(coord)
    return unique_coords_res


def reorder_localize_coords(input_coords, center_x: float, center_y: float):
    """
    Reverse coordinates if polygon is counterclockwise, then translate
    them relative to a given center.

    Parameters
    ----------
    input_coords : LinearRing or Sequence of coordinates
        A shapely LinearRing or any sequence of (x, y) coords.
        Must support `.is_ccw` and `.reverse()`, or adapt as needed.
    center_x : float
        X coordinate to translate from.
    center_y : float
        Y coordinate to translate from.

    Returns
    -------
    list of (float, float)
        The re-ordered, localized (translated) coordinates.
    """
    # If the ring is in CCW order, reverse it so we have consistent winding
    if hasattr(input_coords, "is_ccw") and input_coords.is_ccw:
        input_coords.reverse()

    # Translate coords to local origin at (center_x, center_y)
    res_coords = [
        (coord[0] - center_x, coord[1] - center_y)
        for coord in list(input_coords.coords)
    ]
    return res_coords


def random_building_height(
    building: dict,
    building_polygon: Polygon,  # pylint: disable=unused-argument
) -> float:
    """
    Determine a building's height from OSM tags if available, else random.

    Parameters
    ----------
    building : dict
        A record (row) from an OSM data source containing building attributes,
        e.g. 'building:height', 'height', 'building:levels', etc.
    building_polygon : Polygon
        The polygon geometry of this building (unused in this function's fallback).

    Returns
    -------
    float
        The estimated building height in meters.
    """
    # Seed the random generator based on the building's centroid to ensure consistency
    np.random.seed(
        int(building["geometry"].centroid.x + building["geometry"].centroid.y)
    )
    if "building:height" in building and is_float(building["building:height"]):
        building_height = float(building["building:height"])
    elif "height" in building and is_float(building["height"]):
        building_height = float(building["height"])
    elif "building:levels" not in building or not is_float(building["building:levels"]):
        # Fallback random height (units: meters)
        building_height = 3.5 * max(1, min(15, int(np.random.normal(loc=5, scale=1))))
    elif "level" not in building or not is_float(building["level"]):
        building_height = 3.5 * max(1, min(15, int(np.random.normal(loc=5, scale=1))))
    else:
        building_height = float(building["building:levels"]) * 3.5

    return building_height


def is_float(element) -> bool:
    """
    Check if `element` can be safely cast to a float and is not NaN or inf.

    Parameters
    ----------
    element : any
        The value to check.

    Returns
    -------
    bool
        True if element is a valid float, otherwise False.
    """
    if element is None:
        return False
    try:
        val = float(element)
        return not (math.isnan(val) or math.isinf(val))
    except (TypeError, ValueError):
        return False


def ensure_scene_ready(scene_dir: str, raise_if_missing: bool = True) -> True:
    """Checks that a scene exists.

    If the given path is under `LOCAL_SCENES_DIR` and the scene is not present,
    looks for a corresponding zip file in `REMOTE_SCENES_DIR` and unzips it to
    the local directory.
    """
    if isfile(join(scene_dir, "scene.xml")):
        return True

    if not scene_dir.startswith(LOCAL_SCENES_DIR):
        if not raise_if_missing:
            return False
        raise FileNotFoundError(
            f"Scene not found: {scene_dir}. We did not attempt to unzip it from the remote"
            f" scenes directory because the path is not under {LOCAL_SCENES_DIR}."
        )

    remote_path = join(
        REMOTE_SCENES_DIR, relpath(scene_dir, start=LOCAL_SCENES_DIR) + ".zip"
    )
    if not isfile(remote_path):
        if not raise_if_missing:
            return False
        raise FileNotFoundError(
            f"Scene not found: {scene_dir}. Furthermore, the corresponding zip "
            f"file in the remote scenes directory was not found either: {remote_path}"
        )

    # Unzip the scene to the local directory
    shutil.unpack_archive(remote_path, scene_dir)

    return True


def _extract_scene_worker(
    scenes_parent_dir: str,
    scene_i_and_zip: tuple[int, str],
    raise_if_missing: bool = True,
):
    scene_i, scene_zip = scene_i_and_zip

    scene_dir = join(scenes_parent_dir, scene_zip.replace(".zip", ""))

    try:
        scene_ready = ensure_scene_ready(scene_dir, raise_if_missing=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
        if raise_if_missing:
            raise e

        print(f"[!] Failed to extract {scene_zip}: {repr(e)}")
        return scene_i, None

    if scene_ready:
        result = join(scene_dir, "scene.xml")
    else:
        message = (
            f'Error when trying to prepare scene "{scene_dir}"'
            f' corresponding to archive "{scene_zip}".'
        )
        if raise_if_missing:
            raise FileNotFoundError(message)

        # Missing scene, but we allow it
        print(f"[!] {message}")
        return scene_i, None

    return scene_i, result


def ensure_scenes_ready(
    scenes_parent_dir: str,
    tile_indices: list | None = None,
    progress: bool = False,
    n_processes: int = 1,
    allow_missing: bool = False,
) -> list[str]:
    """
    Ensures that all scenes in the given parent directory are ready.
    The given directory must be under `LOCAL_SCENES_DIR`.
    """
    scenes_parent_dir = realpath(scenes_parent_dir)
    if not scenes_parent_dir.startswith(LOCAL_SCENES_DIR):
        raise ValueError(
            f'The given directory "{scenes_parent_dir}" is not under the'
            f' local scenes directory "{LOCAL_SCENES_DIR}".'
        )

    remote_parent_dir = realpath(
        join(REMOTE_SCENES_DIR, relpath(scenes_parent_dir, start=LOCAL_SCENES_DIR))
    )
    if not isdir(remote_parent_dir):
        raise FileNotFoundError(
            f"Remote scenes directory not found: {remote_parent_dir}"
        )

    if tile_indices is not None:
        zips = [f"{i:08d}.zip" for i in sorted(tile_indices)]
    else:
        zips = [f for f in sorted(os.listdir(remote_parent_dir)) if f.endswith(".zip")]

    with Pool(n_processes) as pool:
        scenes = pool.imap(
            partial(
                _extract_scene_worker,
                scenes_parent_dir,
                raise_if_missing=not allow_missing,
            ),
            enumerate(zips),
            chunksize=min(4, n_processes),
        )
        if progress:
            scenes = tqdm(scenes, desc="Extracting scenes", total=len(zips))
        scenes = list(scenes)

    n_missing = len([fname for _, fname in scenes if fname is None])
    if n_missing > 0:
        message = f"Found {n_missing} / {len(zips)} missing or invalid scenes when extracting zips."
        if allow_missing:
            print(f"[!] {message}")
        else:
            raise FileNotFoundError(message)

    scenes = [fname for _, fname in sorted(scenes, key=lambda x: x[0])]

    tile_corners_local = join(scenes_parent_dir, "bboxes.npz")
    tile_corners_remote = join(remote_parent_dir, "bboxes.npz")
    if isfile(tile_corners_remote) and not isfile(tile_corners_local):
        # Link pointing to (arg 0) named (arg 1)
        os.symlink(tile_corners_remote, tile_corners_local)

    return scenes
