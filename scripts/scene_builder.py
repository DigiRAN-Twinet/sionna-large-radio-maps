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

import mitsuba as mi
import logging
import sys
import numpy as np
import os
import shutil
from argparse import ArgumentParser, RawTextHelpFormatter
import zipfile

import shapely
from shapely.geometry import shape
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import osmnx as ox

from tqdm import tqdm
from triangle import triangulate
from pyproj import Transformer

from common import add_project_root_to_path

add_project_root_to_path()
from sionna_lrm import REMOTE_SCENES_DIR, LOCAL_SCENES_DIR
from sionna_lrm.scene.utils import *
from sionna_lrm.scene.logging_utils import *
from sionna_lrm.scene.heightmap import HeightMap
from sionna_lrm.scene.quadtree import QuadTree


def postprocess_scene(out_dir, scene_name, keep_local):
    if os.path.exists(os.path.join(LOCAL_SCENES_DIR, out_dir, scene_name)):
        shutil.make_archive(
            f"{os.path.join(REMOTE_SCENES_DIR, out_dir, scene_name)}",
            "zip",
            os.path.join(LOCAL_SCENES_DIR, out_dir, scene_name),
        )
        if not keep_local:
            shutil.rmtree(os.path.join(LOCAL_SCENES_DIR, out_dir, scene_name))


def build_scene(
    bbox,
    scene_dir,
    zoom=14,
    osm_server_addr=None,
    compress=True,
    keep_local=False,
    parallel=False,
    overwrite=False,
):
    """
    Generate a ground mesh from the given polygon (defined by `bbox`),
    query OSM for building footprints, extrude them into 3D meshes,
    and optionally produce a 2D building-height map.

    Parameters
    ----------
    bbox : tuple of float
        Bounding box as (min_lon, min_lat, max_lon, max_lat) in WGS84.
    scene_dir : str
        Directory for scene output.
    target_resolution_m : float, optional
        Target resolution in meters for auto-selecting zoom level. Ignored if zoom is provided.
        Defaults to 100.0m if neither zoom nor target_resolution_m are specified.
    osm_server_addr : str, optional
        Custom Overpass API endpoint. If None, osmnx's default is used.
    compress : bool, optional
        Whether to create a zip archive of the scene (default: True).
    keep_local : bool, optional
        Whether to keep the local scene directory after creating zip archive (default: False).
    parallel : bool, optional
        Whether running in parallel mode (default: False).
    overwrite : bool, optional
        Whether to overwrite existing scene directory (default: False).
    """
    mi.set_variant("llvm_ad_rgb")
    # Create a module-level logger
    out_dir, scene_name = os.path.split(scene_dir)
    logger = setup_logging(parallel, scene_name)

    scene_file = os.path.join(LOCAL_SCENES_DIR, out_dir, scene_name, "scene.xml")
    zip_filename = os.path.join(REMOTE_SCENES_DIR, out_dir, f"{scene_name}.zip")
    if not overwrite:
        skip = os.path.exists(scene_file)
        if compress and os.path.exists(zip_filename):
            with zipfile.ZipFile(zip_filename, "r") as f:
                if "scene.xml" in f.namelist():
                    skip = True

        if skip:
            if compress:
                postprocess_scene(out_dir, scene_name, keep_local)
            logger.info(f"Scene {scene_name} already exists, skipping...")
            return

    # Ensure we have enough disk space
    if shutil.disk_usage(LOCAL_SCENES_DIR).free < 10 * 2**30:
        raise RuntimeError("Less than 10GB available disk space, aborting.")

    # ---------------------------------------------------------------------
    # 1) Setup OSM server and transforms
    # ---------------------------------------------------------------------
    if osm_server_addr:
        ox.settings.overpass_url = osm_server_addr
        ox.settings.overpass_rate_limit = False

    # Determine the UTM projection from the first point
    min_lat, min_lon, max_lat, max_lon = bbox
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(
            f"Invalid bounding box coordinates: {bbox}. Coordinates are expected to be in (min_lat, min_lon, max_lat, max_lon) format. If the region of interest crosses the antimeridian, please split it into two separate bounding boxes."
        )
    if min_lat < -90 or max_lat > 90 or min_lon < -180 or max_lon > 180:
        raise ValueError(
            f"Invalid bounding box coordinates: {bbox}. Latitude must be between -90 and 90 degrees, and longitude must be between -180 and 180 degrees."
        )

    logger.info(
        f"Processing this tile: http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}"
    )

    projection_UTM_EPSG_code = get_utm_epsg_code_from_gps(min_lon, min_lat)
    logger.info(f"Using UTM Zone: {projection_UTM_EPSG_code}")

    # Create transformations between WGS84 (EPSG:4326) and UTM
    to_utm = Transformer.from_crs("EPSG:4326", projection_UTM_EPSG_code, always_xy=True)
    to_wgs84 = Transformer.from_crs(
        projection_UTM_EPSG_code, "EPSG:4326", always_xy=True
    )

    # ---------------------------------------------------------------------
    # 2) Prepare output directories and camera / material settings
    # ---------------------------------------------------------------------
    mesh_data_dir = os.path.join(LOCAL_SCENES_DIR, out_dir, scene_name, "mesh")
    os.makedirs(os.path.join(mesh_data_dir), exist_ok=True)

    # Define material colors. This is RGB 0-1 formar https://rgbcolorpicker.com/0-1
    material_colors = {
        "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
        "mat-itu_marble": (0.701101, 0.644479, 0.485150),
        "mat-itu_metal": (0.219526, 0.219526, 0.254152),
        "mat-itu_wood": (0.043, 0.58, 0.184),
        "mat-itu_wet_ground": (0.91, 0.569, 0.055),
        "mat-itu_glass": (0, 0, 1),
    }

    # ---------------------------------------------------------------------
    # 3) Build the XML scene root
    # ---------------------------------------------------------------------

    scene = ET.Element("scene", version="2.1.0")
    bbox_comment = ET.Comment(
        f"http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}"
    )
    scene.append(bbox_comment)

    # Define materials
    for material_id, rgb in material_colors.items():
        bsdf_diffuse = ET.SubElement(scene, "bsdf", type="diffuse", id=material_id)
        ET.SubElement(
            bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance"
        )

    # ---------------------------------------------------------------------
    # 4) Create ground polygon (in UTM) and ground mesh
    # ---------------------------------------------------------------------
    corners = np.array(
        [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]
    )
    coords = np.vstack(to_utm.transform(corners[:, 0], corners[:, 1])).T
    offset = 1e3  # 1km
    extended_coords = coords + np.array(
        [
            [-offset, -offset],
            [offset, -offset],
            [offset, offset],
            [-offset, offset],
            [-offset, -offset],
        ]
    )
    # Convert back to lon/lat to get the extended bbox
    extended_corners = np.vstack(
        to_wgs84.transform(extended_coords[:, 0], extended_coords[:, 1])
    ).T
    ext_min_lon, ext_min_lat = extended_corners.min(axis=0)
    ext_max_lon, ext_max_lat = extended_corners.max(axis=0)
    extended_bbox = (ext_min_lon, ext_min_lat, ext_max_lon, ext_max_lat)
    logger.info(
        f"Extended area: http://bboxfinder.com/#{ext_min_lat:.{4}f},{ext_min_lon:.{4}f},{ext_max_lat:.{4}f},{ext_max_lon:.{4}f}"
    )
    ground_polygon = shapely.geometry.Polygon(extended_coords)

    center_x, center_y = coords[0]  # Set the first corner as origin

    heightmap = HeightMap(
        projection_UTM_EPSG_code, extended_bbox, z=zoom, parallel=parallel
    )
    quadtree = QuadTree(projection_UTM_EPSG_code, bbox, heightmap, parallel=parallel)

    terrain_mesh = quadtree.to_mesh()
    terrain_mesh.write_ply(os.path.join(mesh_data_dir, "ground.ply"))
    material_type = "mat-itu_wet_ground"
    sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
    ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/ground.ply")
    bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
    ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

    for k, bb in enumerate(
        [
            (ext_min_lat, ext_min_lon, min_lat, ext_max_lon),  # Bottom
            (max_lat, ext_min_lon, ext_max_lat, ext_max_lon),  # Top
            (min_lat, ext_min_lon, max_lat, min_lon),  # Left
            (min_lat, max_lon, max_lat, ext_max_lon),  # Right
        ]
    ):
        v, f = bbox_to_mesh(
            bb,
            center=np.array([center_x, center_y, 0.0]),
            heightmap=heightmap,
            to_utm=to_utm,
            resolution=100.0,
        )
        mesh = mi.Mesh(f"ground-buffer-{k}", len(v), len(f))
        params = mi.traverse(mesh)
        params["vertex_positions"] = mi.Float(v.flatten())
        params["faces"] = mi.UInt32(f.flatten())
        params.update()

        mesh.write_ply(os.path.join(mesh_data_dir, f"ground_buffer_{k}.ply"))
        material_type = "mat-itu_wet_ground"
        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground-{k}")
        ET.SubElement(
            sionna_shape, "string", name="filename", value=f"mesh/ground_buffer_{k}.ply"
        )
        bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
        ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

    # ---------------------------------------------------------------------
    # 5) Query OSM for buildings within the bounding box
    # ---------------------------------------------------------------------

    # OSMnx features API uses bounding box in the form (north, south, east, west)
    try:
        buildings = ox.features.features_from_bbox(
            bbox=extended_bbox, tags={"building": True}
        )
        buildings = buildings.to_crs(projection_UTM_EPSG_code)
        # Filter out the building which outside the bounding box since
        # OSM will return some extra buildings.
        filtered_buildings = buildings[buildings.intersects(ground_polygon)]
        buildings_list = filtered_buildings.to_dict("records")
    except ox._errors.InsufficientResponseError:
        logger.info(f"No buildings found")
        buildings_list = []

    # ---------------------------------------------------------------------
    # 8) Process each building to create a 3D mesh (extrude by building height)
    # ---------------------------------------------------------------------

    building_vertices = []
    building_faces = []
    building_vertex_count = 0
    logger.info(f"Processing {len(buildings_list)} buildings...")
    for idx, building in tqdm(
        enumerate(buildings_list),
        total=len(buildings_list),
        desc="Parsing buildings",
        disable=parallel,
    ):
        # Convert building geometry to a shapely polygon
        building_polygon = shape(building["geometry"])

        if building_polygon.geom_type != "Polygon":
            continue

        # First try to get building height from LiDAR
        building_height = random_building_height(building, building_polygon)

        # building_height = NYC_LiDAR_building_height(building, building_polygon)

        outer_xy = unique_coords(
            reorder_localize_coords(building_polygon.exterior, center_x, center_y)
        )
        building_elevation = heightmap.height_from_utm(
            building_polygon.exterior.centroid.x, building_polygon.exterior.centroid.y
        )[0]

        holes_xy = []
        if len(list(building_polygon.interiors)) != 0:
            for inner_hole in list(building_polygon.interiors):
                valid_coords = reorder_localize_coords(inner_hole, center_x, center_y)
                holes_xy.append(unique_coords(valid_coords))

        def edge_idxs(nv):
            i = np.append(np.arange(nv), 0)
            return np.stack([i[:-1], i[1:]], axis=1)

        nv = 0
        verts, edges = [], []
        for loop in (outer_xy, *holes_xy):
            verts.append(loop)
            edges.append(nv + edge_idxs(len(loop)))
            nv += len(loop)

        verts, edges = np.concatenate(verts), np.concatenate(edges)

        # Triangulate needs to know a single interior point for each hole
        # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
        # e.g. shapely's `polylabel`
        holes = np.array([np.mean(h, axis=0) for h in holes_xy])
        # Make sure vertices are unique
        verts, inverse = np.unique(verts, axis=0, return_inverse=True)
        edges = inverse[edges]

        # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
        if len(holes) != 0:
            d = triangulate(dict(vertices=verts, segments=edges, holes=holes), opts="p")
        else:
            d = triangulate(dict(vertices=verts, segments=edges), opts="p")

        # Convert back to pyvista
        v, f = d["vertices"], d["triangles"]
        nv, nf = len(v), len(f)

        points = np.concatenate([v, np.full((nv, 1), building_elevation)], axis=1)
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(points)
        mesh_o3d.triangle.indices = o3d.core.Tensor(f)
        wedge = mesh_o3d.extrude_linear([0, 0, building_height])

        # Add vertices and faces to the global buffer
        building_vertices.append(wedge.vertex.positions)
        building_faces.append(wedge.triangle.indices + building_vertex_count)
        building_vertex_count += wedge.vertex.positions.shape[0]

    # Construct the final mesh from all buildings
    if building_vertex_count > 0:
        if len(building_vertices) > 1:
            combined_vertices = o3d.core.concatenate(building_vertices, axis=0)
            combined_faces = o3d.core.concatenate(building_faces, axis=0)
        else:
            combined_vertices = building_vertices[0]
            combined_faces = building_faces[0]

        buildings_mesh = o3d.t.geometry.TriangleMesh()
        buildings_mesh.vertex.positions = combined_vertices
        buildings_mesh.triangle.indices = combined_faces

        # Save the combined mesh
        o3d.t.io.write_triangle_mesh(
            os.path.join(mesh_data_dir, "buildings.ply"), buildings_mesh
        )

        # Add the combined mesh to the scene
        material_type = "mat-itu_concrete"
        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-buildings")
        ET.SubElement(
            sionna_shape, "string", name="filename", value=f"mesh/buildings.ply"
        )
        bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
        ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

    try:
        water = ox.features.features_from_bbox(
            bbox=bbox,
            tags={
                "water": True,
                "landuse": ["reservoir", "basin", "salt_pond"],
                "natural": ["water"],
            },
        )
        water = water.to_crs(projection_UTM_EPSG_code)
        filtered_water = water[water.intersects(ground_polygon)]
        water_list = filtered_water.to_dict("records")
    except ox._errors.InsufficientResponseError:
        logger.info(f"No buildings found")
        water_list = []

    # ---------------------------------------------------------------------
    # 8) Process each water body to create a 2D mesh
    # ---------------------------------------------------------------------

    water_vertices = []
    water_faces = []
    water_vertex_count = 0
    logger.info(f"Processing {len(water_list)} water bodies...")
    for idx, water in tqdm(
        enumerate(water_list),
        total=len(water_list),
        desc="Parsing water",
        disable=parallel,
    ):
        water_polygon = shape(water["geometry"])

        if water_polygon.geom_type != "Polygon":
            continue

        outer_xy = unique_coords(
            reorder_localize_coords(water_polygon.exterior, center_x, center_y)
        )
        water_elevation = (
            heightmap.height_from_utm(
                water_polygon.exterior.centroid.x, water_polygon.exterior.centroid.y
            )[0]
            + 0.5
        )  # Add a small offset to avoid z-fighting with the ground mesh

        holes_xy = []
        if len(list(water_polygon.interiors)) != 0:
            for inner_hole in list(water_polygon.interiors):
                valid_coords = reorder_localize_coords(inner_hole, center_x, center_y)
                holes_xy.append(unique_coords(valid_coords))

        def edge_idxs(nv):
            i = np.append(np.arange(nv), 0)
            return np.stack([i[:-1], i[1:]], axis=1)

        nv = 0
        verts, edges = [], []
        for loop in (outer_xy, *holes_xy):
            verts.append(loop)
            edges.append(nv + edge_idxs(len(loop)))
            nv += len(loop)

        verts, edges = np.concatenate(verts), np.concatenate(edges)
        # Make sure vertices are unique
        verts, inverse = np.unique(verts, axis=0, return_inverse=True)
        edges = inverse[edges]

        # Triangulate needs to know a single interior point for each hole
        # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
        # e.g. shapely's `polylabel`
        holes = np.array([np.mean(h, axis=0) for h in holes_xy])

        # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
        if len(holes) != 0:
            d = triangulate(dict(vertices=verts, segments=edges, holes=holes), opts="p")
        else:
            d = triangulate(dict(vertices=verts, segments=edges), opts="p")

        # Convert back to pyvista
        v, f = d["vertices"], d["triangles"]
        nv, nf = len(v), len(f)

        points = np.concatenate([v, np.full((nv, 1), water_elevation)], axis=1)
        # Add vertices and faces to the global buffer
        water_vertices.append(o3d.core.Tensor(points))
        water_faces.append(o3d.core.Tensor(f) + water_vertex_count)
        water_vertex_count += points.shape[0]

    if water_vertex_count > 0:
        if len(water_vertices) > 1:
            combined_vertices = o3d.core.concatenate(water_vertices, axis=0)
            combined_faces = o3d.core.concatenate(water_faces, axis=0)
        else:
            combined_vertices = water_vertices[0]
            combined_faces = water_faces[0]

        water_mesh = o3d.t.geometry.TriangleMesh()
        water_mesh.vertex.positions = combined_vertices
        water_mesh.triangle.indices = combined_faces

        # Save the combined mesh
        o3d.t.io.write_triangle_mesh(
            os.path.join(mesh_data_dir, "water.ply"), water_mesh
        )

        material_type = "mat-itu_glass"
        # Add shape elements for PLY files in the folder
        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-water")
        ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/water.ply")
        bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
        ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

    xml_string = ET.tostring(scene, encoding="utf-8")
    xml_pretty = minidom.parseString(xml_string).toprettyxml(
        indent="    "
    )  # Adjust the indent as needed

    with open(scene_file, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_pretty)

    # Make an archive in the remote scenes directory
    if compress:
        postprocess_scene(out_dir, scene_name, keep_local)

    logger.info(f"Scene saved to {os.path.join(LOCAL_SCENES_DIR, out_dir, scene_name)}")


def build_scene_wrapper(params):
    (
        bbox,
        scene_dir,
        zoom,
        osm_server_addr,
        compress,
        keep_local,
        parallel,
        idx,
        overwrite,
    ) = params
    _, scene_name = os.path.split(scene_dir)
    logger = setup_logging(parallel, scene_name)
    try:
        build_scene(
            bbox,
            scene_dir,
            zoom,
            osm_server_addr,
            compress,
            keep_local,
            parallel,
            overwrite,
        )
        return idx, True
    except Exception as e:
        logger.exception(f"Error processing region {idx}: {e}")
        return idx, False


if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")

    parser = ArgumentParser(
        description="Scene Generation CLI.\n\n"
        "You can define the scene location (a rectangle) in two ways:\n"
        "  1) 'bbox' subcommand: specify four GPS corners (south, west, north, east).\n"
        "  2) 'point' subcommand: specify one GPS point, indicate its corner/center position, "
        "  3) 'file' subcommand: specify a file with a list of bboxes, "
        "and give width/height in meters.\n",
        formatter_class=RawTextHelpFormatter,
    )

    # --version/-v: we'll handle printing version info ourselves after parse_args()
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version information and exit.",
    )

    # Create a "parent" parser to hold common optional arguments.
    # Use add_help=False so we don’t duplicate the --help in child parsers.
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output."
    )
    common_parser.add_argument(
        "--osm-server-addr",
        default="https://overpass-api.de/api/interpreter",
        help="(Optional) OSM server address.",
    )
    common_parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "If passed, sets console logging to DEBUG (file logging is always DEBUG). "
            "This overrides the default console level of INFO."
        ),
    )
    common_parser.add_argument(
        "--zoom",
        type=int,
        default=14,
        help="Slippy map zoom level for heightmap tiles (default: 14).",
    )

    common_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If passed, overwrite existing scene directories. Otherwise, existing scenes will be skipped.",
    )

    common_parser.add_argument(
        "--keep-local",
        action="store_true",
        help="If passed, keep local scene directory after creating zip archive.",
    )

    common_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="If passed, do not create a zip archive of each scene.",
    )

    # Create subparsers for different subcommands
    subparsers = parser.add_subparsers(
        title="Subcommands", dest="command", help="Available subcommands."
    )

    # Subcommand 'bbox': define a bounding box by four float coordinates
    parser_bbox = subparsers.add_parser(
        "bbox",
        parents=[common_parser],
        help=(
            "Define a bounding box using four GPS coordinates in the order: "
            "south west north east"
        ),
    )
    parser_bbox.add_argument("min_lat", type=float, help="Minimum latitude.")
    parser_bbox.add_argument("min_lon", type=float, help="Minimum longitude.")
    parser_bbox.add_argument("max_lat", type=float, help="Maximum latitude.")
    parser_bbox.add_argument("max_lon", type=float, help="Maximum longitude.")
    parser_bbox.add_argument(
        "--scene-name",
        required=True,
        help="Directory where scene file will be saved.",
    )

    # Subcommand 'point': define a reference point and rectangle size
    parser_point = subparsers.add_parser(
        "point",
        parents=[common_parser],
        help="Work with a single point and a rectangle size.",
    )
    parser_point.add_argument("lat", type=float, help="Latitude.")
    parser_point.add_argument("lon", type=float, help="Longitude.")
    parser_point.add_argument(
        "height", type=float, help="Area size along the latitude axis in meters."
    )
    parser_point.add_argument(
        "width", type=float, help="Area size along the longitude axis in meters."
    )
    parser_point.add_argument(
        "--position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        default="bottom-left",
        help="Relative position inside a rectangle.",
    )
    parser_point.add_argument(
        "--scene-name",
        required=True,
        help="Directory where scene file will be saved.",
    )

    parser_file = subparsers.add_parser(
        "file",
        parents=[common_parser],
        help="Process a list of bounding boxes from a file.",
    )
    parser_file.add_argument(
        "file",
        type=str,
        help="Path to a NPZ file containing bounding boxes in the 'corners' field.",
    )

    parser_file.add_argument(
        "--n",
        type=int,
        default=16,
        help="Number of parallel processes to use for scene generation.",
    )

    parser_file.add_argument(
        "--tiles",
        type=str,
        help="(Optional) .npy file containing indices of tiles to process.",
    )

    parser_file.add_argument(
        "--tile", type=int, help="(Optional) Index of a single tile to process."
    )

    parser_file.add_argument(
        "--subdir",
        type=str,
        default="",
        help="(Optional) Subdirectory inside scenes/ to save the scenes.",
    )

    # Parse the full command line
    args = parser.parse_args()

    if not args.command:
        # No subcommand provided: show help and exit
        parser.print_help()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1) Set up logging by default
    #    - debug.log file captures all logs at DEBUG level
    #    - console sees INFO+ by default
    # -------------------------------------------------------------------------
    logger = setup_logging(False)

    # If user wants console debug output too, adjust console handler level.
    if args.debug:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)

    # Dispatch subcommands
    if args.command == "bbox":
        build_scene(
            (args.min_lat, args.min_lon, args.max_lat, args.max_lon),
            args.scene_name,
            osm_server_addr=args.osm_server_addr,
            compress=False,
            zoom=args.zoom,
        )
    elif args.command == "point":
        polygon_points_gps = rect_from_point_and_size(
            args.lon, args.lat, args.position, args.width, args.height
        )
        min_lon, min_lat = polygon_points_gps[0]
        max_lon, max_lat = polygon_points_gps[2]
        build_scene(
            (min_lat, min_lon, max_lat, max_lon),
            args.scene_name,
            osm_server_addr=args.osm_server_addr,
            compress=False,
            zoom=args.zoom,
            overwrite=args.overwrite,
        )
    elif args.command == "file":
        bboxes = np.load(args.file)
        print(f"Loaded {len(bboxes['corners'])} bounding boxes from {args.file}")
        # Save the NPZ file to the scene directory for reference
        os.makedirs(os.path.join(REMOTE_SCENES_DIR, args.subdir), exist_ok=True)
        np.savez(os.path.join(REMOTE_SCENES_DIR, args.subdir, "bboxes.npz"), **bboxes)
        os.makedirs(os.path.join(LOCAL_SCENES_DIR, args.subdir), exist_ok=True)
        np.savez(os.path.join(LOCAL_SCENES_DIR, args.subdir, "bboxes.npz"), **bboxes)

        from multiprocessing import Pool, set_start_method

        set_start_method("spawn")
        scene_params = []
        if args.tiles is not None:
            tile_idx = np.load(args.tiles)
            tiles = bboxes["corners"][tile_idx]
        elif args.tile is not None:
            tiles = [bboxes["corners"][args.tile]]
        else:
            tiles = bboxes["corners"]
            tile_idx = np.arange(len(tiles))

        parallel = (args.n > 1) and (len(tiles) > 1)
        if not parallel:
            for i, bbox in enumerate(tiles):
                if args.tile:
                    name = f"{args.tile:08d}"
                elif args.tiles:
                    name = f"{tile_idx[i]:08d}"
                else:
                    name = f"{i:08d}"
                build_scene(
                    bbox.flatten(),
                    os.path.join(args.subdir, name),
                    args.zoom,
                    args.osm_server_addr,
                    not args.no_compress,
                    args.keep_local,
                    False,  # parallel
                    overwrite=args.overwrite,
                )
        else:
            np.random.seed(0)
            np.random.shuffle(tile_idx)
            for i in tile_idx:
                scene_params.append(
                    (
                        tiles[i].flatten(),
                        os.path.join(args.subdir, f"{i:08d}"),
                        args.zoom,
                        args.osm_server_addr,
                        not args.no_compress,
                        args.keep_local,
                        parallel,
                        i,
                        args.overwrite,
                    )
                )

            with Pool(args.n) as pool:
                # Use tqdm to show progress bar
                result = list(
                    tqdm(
                        pool.imap_unordered(
                            build_scene_wrapper,
                            scene_params,
                            chunksize=1,
                        ),
                        total=len(scene_params),
                        desc="Processing scenes",
                    )
                )
                success = np.array(result)[:, 1].astype(bool)
                idx = np.array(result)[:, 0].astype(int)
                if np.any(~success):
                    failed_tiles = idx[~success]
                    filename = os.path.join(
                        LOCAL_SCENES_DIR, args.subdir, "failed_tiles.npy"
                    )
                    np.save(filename, failed_tiles)
                    print(
                        f"{len(failed_tiles)} scenes failed. Indices saved to {filename}."
                    )
                else:
                    print("All scenes processed successfully.")
    else:
        # Should never happen if we covered all subcommands
        parser.print_help()
        sys.exit(1)
