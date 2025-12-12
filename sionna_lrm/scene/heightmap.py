#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import drjit as dr
import mitsuba as mi
import numpy as np
from PIL import Image
from pyproj import Transformer
from tqdm import tqdm

from .logging_utils import setup_logging


class HeightMap:
    def __init__(self, utm_zone, bbox, z=14, parallel=False):
        self.s3_client = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=100,
                region_name="us-east-1",
                retries={"max_attempts": 2, "mode": "adaptive"},
            ),
        )

        self.logger = setup_logging(parallel)
        self.bbox_min_lon, self.bbox_min_lat, self.bbox_max_lon, self.bbox_max_lat = (
            bbox
        )
        x0 = int(self.lon2tile(self.bbox_min_lon, z))
        x1 = int(self.lon2tile(self.bbox_max_lon, z))
        y0 = int(self.lat2tile(self.bbox_max_lat, z))
        y1 = int(self.lat2tile(self.bbox_min_lat, z))

        self.to_utm = Transformer.from_crs("EPSG:4326", utm_zone, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)

        n_rows = y1 - y0 + 1
        n_cols = x1 - x0 + 1
        tex = np.zeros((n_rows * 512, n_cols * 512, 1))
        self.logger.info(
            f"Loading heightmap tiles from {x0}, {y0} to {x1}, {y1} at zoom level {z}"
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.get_tile, z, x0 + col, y0 + row)
                for row in range(n_rows)
                for col in range(n_cols)
            ]
            for future in tqdm(
                futures, desc="Downloading tiles", total=len(futures), disable=parallel
            ):
                x, y, tile = future.result()
                col = x - x0
                row = y - y0
                tex[row * 512 : (row + 1) * 512, col * 512 : (col + 1) * 512, 0] = tile

        self.heightmap = mi.load_dict(
            {
                "type": "bitmap",
                "data": mi.TensorXf(tex),
                "raw": True,
                "filter_type": "bilinear",
            }
        )

        # heightmap covers the area of all tiles, not just our bbox
        self.map_min_lon = self.tile2lon(x0, z)
        self.map_min_lat = self.tile2lat(y0, z)
        self.map_max_lon = self.tile2lon(x1 + 1, z)
        self.map_max_lat = self.tile2lat(y1 + 1, z)

    def download_tile(self, z, x, y):
        key = f"geotiff/{z}/{x}/{y}.tif"
        buffer = BytesIO()
        self.s3_client.download_fileobj("elevation-tiles-prod", key, buffer)
        buffer.seek(0)
        im = np.array(Image.open(buffer), dtype=np.float32)
        if im.shape != (512, 512):
            # Resample using mitsuba
            im = (
                mi.TensorXf(mi.Bitmap(im).resample(mi.ScalarVector2u(512)))
                .numpy()
                .squeeze()
            )
        return im

    def get_tile(self, z, x, y):
        im = self.download_tile(z, x, y)

        i = 0
        while np.any((im < -1e6) | np.isnan(im) | (im > 1e4)):
            i += 1
            self.logger.warning(
                f"Tile {x},{y} at zoom {z} contains invalid pixels,"
                f" filling them in using zoom level {z - i}..."
            )
            if i == 10:
                self.logger.error(
                    f"Cannot fill invalid pixels for tile {x},{y}, reached minimum zoom level."
                )
                return x, y, np.zeros((512, 512), dtype=np.float32)

            # Compute parent tile coordinates
            scale = 2**i
            parent_tile = self.download_tile(z - i, x // scale, y // scale)
            # Extract slice of parent tile corresponding to current tile
            im = parent_tile[
                (y % scale) * 512 // scale : (y % scale + 1) * 512 // scale,
                (x % scale) * 512 // scale : (x % scale + 1) * 512 // scale,
            ]

        if im.shape != (512, 512):
            # Resample using mitsuba
            im = (
                mi.TensorXf(mi.Bitmap(im).resample(mi.ScalarVector2u(512)))
                .numpy()
                .squeeze()
            )

        return x, y, im

    @staticmethod
    def lat2tile(lat, z):
        lat_rad = np.deg2rad(lat)
        n = 2**z
        yt = np.arcsinh(np.tan(lat_rad))
        return np.floor(n * 0.5 * (1 - yt / np.pi))

    @staticmethod
    def lon2tile(lon, z):
        lon_rad = np.deg2rad(lon)
        n = 2**z
        return np.floor(n * 0.5 * (1 + lon_rad / np.pi))

    @staticmethod
    def tile2lon(x, z):
        n = 2**z
        return x / n * 360 - 180

    @staticmethod
    def tile2lat(y, z):
        n = 2**z
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        return np.rad2deg(lat_rad)

    def height_from_utm(self, x, y):
        x_wgs, y_wgs = self.to_wgs84.transform(x, y)
        return self.height_from_wgs84(x_wgs, y_wgs)

    def height_from_wgs84(self, x, y):
        # Normalize the coordinates to [0, 1]
        u = (x - self.map_min_lon) / (self.map_max_lon - self.map_min_lon)
        v = (y - self.map_min_lat) / (self.map_max_lat - self.map_min_lat)
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.uv = mi.Point2f(mi.Float(u), mi.Float(v))
        dr.make_opaque(si)
        return self.heightmap.eval_1(si)

    def generate_terrain(self, n=10**3):
        bottom_left = self.to_utm.transform(self.bbox_min_lon, self.bbox_min_lat)
        top_right = self.to_utm.transform(self.bbox_max_lon, self.bbox_max_lat)

        bbox_utm_center = mi.ScalarPoint2f(
            0.5 * (bottom_left[0] + top_right[0]), 0.5 * (bottom_left[1] + top_right[1])
        )
        bbox_utm_extents = mi.ScalarPoint2f(
            top_right[0] - bottom_left[0], top_right[1] - bottom_left[1]
        )

        xx, yy = dr.meshgrid(
            dr.linspace(
                mi.Float, -0.5 * bbox_utm_extents.x, 0.5 * bbox_utm_extents.x, n
            ),
            dr.linspace(
                mi.Float, -0.5 * bbox_utm_extents.y, 0.5 * bbox_utm_extents.y, n
            ),
            indexing="ij",
        )

        # Get height from the heightmap
        height = self.height_from_utm(
            bbox_utm_center.x + xx.numpy(), bbox_utm_center.y + yy.numpy()
        )

        # Create vertices and faces for the terrain mesh
        verts = mi.Point3f(xx, yy, height)
        faces = dr.zeros(mi.UInt32, 3 * 2 * (n - 1) ** 2)
        idx_base = dr.arange(mi.UInt32, n - 1)
        i = dr.tile(idx_base, n - 1) + dr.repeat(idx_base, n - 1) * n
        j = i + 1
        k = j + n
        f1 = mi.Vector3u(i, j, k)
        f2 = mi.Vector3u(i, k, k - 1)
        idx = dr.arange(mi.UInt32, 3 * (n - 1) ** 2)
        dr.scatter(faces, dr.ravel(f1), idx)
        dr.scatter(faces, dr.ravel(f2), idx + 3 * (n - 1) ** 2)

        terrain_mesh = mi.Mesh("terrain", n**2, 2 * (n - 1) ** 2)
        params = mi.traverse(terrain_mesh)
        params["vertex_positions"] = dr.ravel(verts)
        params["faces"] = faces
        params.update()
        return terrain_mesh
