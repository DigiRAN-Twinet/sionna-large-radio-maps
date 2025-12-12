#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from argparse import ArgumentParser
from contextlib import nullcontext
from functools import partial
from multiprocessing import Pool, Manager, Queue, set_start_method
import os
import traceback

import drjit as dr
import mitsuba as mi
import numpy as np
from pyproj import Transformer
from tqdm import tqdm

from common import add_project_root_to_path

add_project_root_to_path()

from sionna_lrm import SLRM_OPTIX_CACHE_PATH
from sionna_lrm.base_stations import BaseStationDB
from sionna_lrm.constants import DEFAULT_TRANSMITTERS_FNAME
from sionna_lrm.rm_utils import get_highest_at_positions
from sionna_lrm.scene.utils import get_utm_epsg_code_from_gps

# See https://github.com/mitsuba-renderer/mitsuba3/issues/892#issuecomment-1918652252
os.environ["OPTIX_CACHE_PATH"] = SLRM_OPTIX_CACHE_PATH


def process_tile(
    bbox: tuple[tuple[float, float], tuple[float, float]],
    scene_file: str,
    tx_db: BaseStationDB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the elevations for all base stations in the given bbox by loading the associated
    scene and querying the geometry.
    """
    # Note: setting the variant here to allow this function to run with multiprocessing.
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

    (min_lat, min_lon), (max_lat, max_lon) = bbox
    projection_utm_epsg_code = get_utm_epsg_code_from_gps(min_lon, min_lat)
    to_utm = Transformer.from_crs("EPSG:4326", projection_utm_epsg_code, always_xy=True)
    # Set this corner as the origin point
    center_x, center_y = to_utm.transform(min_lon, min_lat)

    # Find transmitters that belong in this tile
    region_db = tx_db.get_region(
        (min_lat, min_lon), (max_lat, max_lon), restrict_to_latlon_bbox=True
    )
    if region_db.tx_count == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    # Compute elevation as the highest point in the scene at the transmitter (x, y)
    mi_scene = mi.load_file(scene_file, optimize=False)
    building_ptr = None
    for shape in mi_scene.shapes():
        if shape.id() == "mesh-buildings":
            building_ptr = mi.ShapePtr(shape)
            break

    tx_latlon = region_db.latlon()
    tx_utm = np.vstack(to_utm.transform(tx_latlon[:, 1], tx_latlon[:, 0])).T
    pts = mi.Point2f(tx_utm.T) - mi.Vector2f(center_x, center_y)
    dr.make_opaque(pts)
    si = get_highest_at_positions(
        mi_scene, pts, allow_miss=True, fallback_to_scene_max=False
    )
    is_valid = si.is_valid().numpy()

    over_building = (si.shape == building_ptr).numpy()[is_valid]
    valid_heights = si.p.z.numpy()[is_valid]
    valid_idx = region_db.tx_df.index.to_numpy()[is_valid]

    return valid_idx, valid_heights, over_building


def loop_body(
    scenes_dir: str,
    tx_db: BaseStationDB,
    tile_i_and_bbox: tuple[int, tuple[tuple[float, float], tuple[float, float]]],
    gpu_queue: Queue | None = None,
) -> tuple[int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if gpu_queue is not None:
        # If not already done, try and get the next free GPU index and use that.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            try:
                my_gpu: int = gpu_queue.get_nowait()
            except ValueError:
                my_gpu = 0

            os.environ["CUDA_VISIBLE_DEVICES"] = str(my_gpu)
            os.environ["OPTIX_CACHE_PATH"] = SLRM_OPTIX_CACHE_PATH
            print(f"[i] Process {os.getpid()} using GPU {my_gpu}.")

    tile_i, bbox = tile_i_and_bbox
    scene_file = os.path.join(scenes_dir, f"{tile_i:08d}", "scene.xml")

    if not os.path.exists(scene_file):
        print(f"[!] Scene file not found for tile {tile_i:08d}: {scene_file}")
        return tile_i, None, None, None

    try:
        valid_idx_i, valid_heights_i, over_building_i = process_tile(
            bbox, scene_file, tx_db
        )
        return tile_i, valid_idx_i, valid_heights_i, over_building_i

    except Exception as e:
        print(
            f"[!] Error processing tile {tile_i:08d}: {repr(e)}\n"
            + traceback.format_exc()
        )
        return tile_i, None, None, None


def serialized_imap(func, iterable, chunksize: int = 1):
    for v in iterable:
        yield func(v)


def main(
    scenes: str,
    transmitters: str,
    bboxes: str | None = None,
    output_fname: str | None = None,
    n_processes: int = 1,
    n_gpus: int = 1,
    only_tiles_i: set[int] | None = None,
    region: list[float] | None = None,
) -> bool:
    if bboxes is None:
        bboxes = os.path.join(scenes, "bboxes.npz")
    if output_fname is None:
        output_fname = transmitters

    tiles = np.load(bboxes)["corners"]
    numbered_tiles = {i: v for i, v in enumerate(tiles)}
    if only_tiles_i is not None:
        only_tiles_i = set(only_tiles_i)
        numbered_tiles = {i: v for i, v in numbered_tiles.items() if i in only_tiles_i}
    if region is not None:
        south, west, north, east = region

        def bbox_intersects(bbox):
            (min_lat, min_lon), (max_lat, max_lon) = bbox
            return not (
                max_lat < south or min_lat > north or max_lon < west or min_lon > east
            )

        numbered_tiles = {i: v for i, v in numbered_tiles.items() if bbox_intersects(v)}

    tx_db = BaseStationDB.from_file(transmitters)

    # Launch processing in parallel
    # Switch to a plain loop if a single process is used, which makes debugging easier.
    use_multiprocessing = n_processes > 1
    print(
        f"[i] Processing {len(numbered_tiles)} tiles with {n_processes} job{'s' if use_multiprocessing else ''}"
    )
    with Manager() if use_multiprocessing else nullcontext() as manager:
        gpu_queue = None
        if use_multiprocessing:
            # Assign a different GPU index to each process
            gpu_queue = manager.Queue()
            for process_i in range(n_processes):
                gpu_queue.put(process_i % n_gpus)

        with Pool(n_processes) if use_multiprocessing else nullcontext() as pool:
            results = list(
                tqdm(
                    (pool.imap if use_multiprocessing else serialized_imap)(
                        partial(loop_body, scenes, tx_db, gpu_queue=gpu_queue),
                        list(numbered_tiles.items()),
                        chunksize=4,
                    ),
                    total=len(numbered_tiles),
                    desc="Processing tiles",
                )
            )

    # Extract results
    valid_idx = []
    valid_heights = []
    over_building = []
    tile_sources = []
    ignored_tiles = []
    for r in results:
        tile_i, valid_idx_i, valid_heights_i, over_building_i = r
        if valid_idx_i is None:
            ignored_tiles.append(tile_i)
            continue
        valid_idx.extend(valid_idx_i)
        valid_heights.extend(valid_heights_i)
        over_building.extend(over_building_i)
        tile_sources.extend([tile_i] * len(valid_idx_i))

    if ignored_tiles:
        print(
            f"[!] Ignored {len(ignored_tiles)} / {len(numbered_tiles)} tiles with no scene."
        )
    if not valid_idx:
        print("[!] No valid tiles found, returning.")
        return False

    # Validate the collected elevations: if a given transmitter appears several
    # times, the heights should be consistent.
    valid_idx = np.array(valid_idx)
    valid_heights = np.array(valid_heights)
    tile_sources = np.array(tile_sources)
    over_building = np.array(over_building)

    progress = tqdm(valid_idx, desc="Validating consistency")
    inconsistent_tx = set()
    for i, tx_idx in enumerate(progress):
        if tx_idx in inconsistent_tx:
            # Already checked this one.
            continue

        same_idx = valid_idx == tx_idx
        if len(same_idx) == 1:
            # No duplicates, so no consistency check needed.
            continue

        # Note: absolute tolerance in meters.
        if not np.allclose(valid_heights[same_idx], valid_heights[i], atol=0.5):
            progress.write(
                f"[!] Transmitter {tx_idx} appears multiple times with different heights:"
                f" {valid_heights[same_idx]}, from tiles {tile_sources[same_idx]}."
            )
            progress.write(
                "   BBoxes of tiles (lower left latlon, upper right latlon):"
            )
            for tile_idx in np.unique(tile_sources[same_idx]):
                progress.write(f"   - Tile {tile_idx:08d}: {numbered_tiles[tile_idx]}")
            inconsistent_tx.add(tx_idx)

    progress.close()

    if inconsistent_tx:
        print(
            f"[!] Found {len(inconsistent_tx)} / {len(valid_idx)} transmitters with"
            " inconsistent elevations. Will not write to the output file."
        )
        return False

    # Update the transmitters dataset with the collected elevations
    _, unique_idx = np.unique(valid_idx, return_index=True)
    valid_idx = valid_idx[unique_idx]
    valid_heights = valid_heights[unique_idx]

    tx_db.set_elevation(valid_idx, valid_heights)
    tx_db.set_over_building(valid_idx, over_building[unique_idx])
    tx_db.to_file(output_fname)

    print(f"[+] Wrote updated dataset with elevations to: {output_fname}")

    return True


if __name__ == "__main__":
    parser = ArgumentParser(
        "Update the dataset of base stations with their elevation, in meters,"
        " computed as the highest point in their corresponding tile (scene)."
        " Note that a small offset should likely be added to the elevation when"
        " using the transmitters."
    )
    parser.add_argument(
        "--transmitters",
        "-t",
        type=str,
        help=f'Path to the transmitters file. Will use "{DEFAULT_TRANSMITTERS_FNAME}" by default.',
        default=DEFAULT_TRANSMITTERS_FNAME,
    )
    parser.add_argument(
        "--bboxes",
        "-b",
        type=str,
        help="Path to the NPZ file containing bounding boxes. By default, will use `bboxes.npz` under the scenes directory.",
        default=None,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output CSV file to write the augmented transmitters dataset to. ",
        default=None,
        dest="output_fname",
    )
    parser.add_argument(
        "--processes",
        "-j",
        type=int,
        help="How many processes to use for parallel processing.",
        default=8,
        dest="n_processes",
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=int,
        help="How many GPUs to use for parallel processing. The number of GPUs actually used cannot be larger than the number of processes (-j).",
        default=1,
        dest="n_gpus",
    )
    parser.add_argument(
        "scenes",
        type=str,
        help="Path to the directory containing the scenes (tiles)."
        " Use the local directory with unzipped scenes, not the"
        " remote directory containing the zip files.",
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=None,
        help="Bounding box as [south west north east] (in degrees) to which processing should be restricted.",
    )

    args = parser.parse_args()
    set_start_method("spawn")
    main(**vars(args))
