#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import gc
import os
import time
from typing import Callable

import drjit as dr
import mitsuba as mi
import numpy as np
from pyproj import Transformer
from tqdm.auto import tqdm
import sionna.rt as rt

from . import antenna_patterns as _  # noqa
from .base_stations import BaseStationDB
from .constants import (
    DEFAULT_LATLON_CRS,
    DEFAULT_MEASUREMENT_MESH_NAME,
    DEFAULT_TX_SEARCH_DISTANCE_M,
    ASSUMED_SCENE_SIZE_MIB,
    MIN_SAMPLES_PER_TX,
    DEFAULT_ANTENNA_ARRAY_PARAMS,
    DEFAULT_Z_OFFSET_BUILDING,
    DEFAULT_Z_OFFSET_NON_BUILDING,
)
from .scene.utils import ensure_scenes_ready, get_utm_epsg_code_from_gps
from .rm_utils import (
    get_highest_at_positions,
    split_work_into_passes,
    get_gpu_available_memory_mib,
)


def get_tx_positions(
    scene: rt.Scene,
    tx_utm: np.ndarray,
    tx_z: np.ndarray,
    tx_is_building: np.ndarray,
    tx_z_offset_building: float,
    tx_z_offset_non_building: float,
) -> tuple[np.ndarray, int]:
    missing_elevations = np.isnan(tx_z)
    tx_pos_np = np.concatenate([tx_utm, tx_z[:, None]], axis=1)

    n_missing = np.sum(missing_elevations)
    if n_missing > 0:
        tx_xy = mi.Point2f(tx_utm.T)
        dr.make_opaque(tx_xy)
        tx_si = get_highest_at_positions(
            scene.mi_scene,
            tx_xy,
            fallback_to_scene_max=True,
            allow_miss=True,
        )
        tx_pos_np[missing_elevations, 2] = tx_si.p.z.numpy()[missing_elevations]

    tx_pos_np[:, 2] += np.where(
        tx_is_building, tx_z_offset_building, tx_z_offset_non_building
    )
    # Return as (3, n_transmitters) for better compatibility with Mitsuba's SoA layout.
    return tx_pos_np.T, n_missing


def estimate_max_rm_entries_per_pass(
    gpu_i: int | None = None,
    assumed_scene_size_mib: int = ASSUMED_SCENE_SIZE_MIB,
) -> tuple[int, int, int]:
    """Estimates the maximum number of radio map entries per pass based on the available memory
    and the desired minimum number of samples per transmitter.

    If `gpu_i` is not provided, we will try reading `CUDA_VISIBLE_DEVICES` from the environment.
    If not set, then GPU 0 is used.
    """
    vram_mib = get_gpu_available_memory_mib(gpu_i)
    available_bytes = (1024 * 1024) * (vram_mib - assumed_scene_size_mib)
    max_entries = max(1, available_bytes // 4)

    # Also, DrJit does not support arrays with more than 2^32 entries.
    max_entries = min(max_entries, 2**32)

    return max_entries, vram_mib


def compute_rm_for_tile(
    scene: rt.Scene,
    tx_utm: np.ndarray,
    local_tx_db: BaseStationDB,
    measurement_surface_id: str,
    seed: int,
    tile_i: int,
    n_samples: int,
    max_rm_entries_per_pass: int,
    max_depth: int = 5,
    tx_z_offset_building: float | None = None,
    tx_z_offset_non_building: float | None = None,
    measurement_z_offset: float = 1.5,
    writer: Callable[[str], None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if writer is None:
        writer = print
    if tx_z_offset_building is None:
        tx_z_offset_building = DEFAULT_Z_OFFSET_BUILDING
    if tx_z_offset_non_building is None:
        tx_z_offset_non_building = DEFAULT_Z_OFFSET_NON_BUILDING

    n_samples = int(n_samples)

    # 1. Set up the transmitters.
    scene.tx_array = rt.PlanarArray(**DEFAULT_ANTENNA_ARRAY_PARAMS)
    original_surface = scene.objects[measurement_surface_id.strip("mesh-")]

    # In principle, the elevation of most transmitters should have been
    # pre-computed, but we can still raycast again if needed.
    n_transmitters = tx_utm.shape[0]
    tx_pos_np, n_missing_elevations = get_tx_positions(
        scene,
        tx_utm,
        local_tx_db.elevation(),
        tx_is_building=local_tx_db.is_over_building(),
        tx_z_offset_building=tx_z_offset_building,
        tx_z_offset_non_building=tx_z_offset_non_building,
    )
    if n_missing_elevations > 0:
        writer(
            f"[!] Tile {tile_i:08d}: found {n_missing_elevations} / {n_transmitters}"
            " transmitters with missing elevation data, backfilled with raycasting."
        )

    if n_transmitters == 0:
        # No transmitters, just return a zero radio map.
        writer(f"[!] Tile {tile_i:08d}: no transmitters, returning zero radio map.")
        n_faces = original_surface.mi_mesh.face_count()
        return np.zeros(shape=(n_faces,), dtype=np.float32), tx_pos_np

    # 2. Prepare mesh-based radio map solver.
    solver = rt.RadioMapSolver()

    # 3. Prepare measurement surface
    surface = original_surface.clone(as_mesh=True)
    del original_surface
    rt.transform_mesh(surface, translation=[0, 0, measurement_z_offset])

    # 4. Determine number of passes based on estimated memory usage and sample count.
    n_passes, n_tx_per_pass, n_samples_per_tx = split_work_into_passes(
        n_transmitters,
        surface.face_count(),
        n_samples,
        max_rm_entries_per_pass,
        MIN_SAMPLES_PER_TX,
    )

    if n_passes > 50:
        writer(
            f"[!] Tile {tile_i:08d}: needs {n_passes} passes due to"
            f" samples count {n_samples}, and face count {surface.face_count()}."
            f" Consider adjusting the module-level constants based on available GPU memory."
        )

    # 5. Compute radio map (in multiple passes if needed)
    rm_max_path_gain = None
    transmitters_used = []
    for pass_i in range(n_passes):
        # Add transmitters for this pass
        scene._transmitters.clear()  # pylint: disable=protected-access
        for tx_i in range(
            pass_i * n_tx_per_pass, min(n_transmitters, (pass_i + 1) * n_tx_per_pass)
        ):
            global_tx_idx = local_tx_db.index_at(tx_i)
            tx = rt.Transmitter(
                f"tx-{global_tx_idx:04d}",
                tx_pos_np[:, tx_i],
            )

            # We seed an RNG with the unique transmitter index so that the
            # same orientation is applied if the same transmitter is used in multiple tiles.
            rng = np.random.default_rng(global_tx_idx)
            vertical_rotation = rng.uniform(-np.pi, np.pi)
            # Note: tilt is applied electronically when using a triplanar array.
            tx.orientation = np.array((vertical_rotation, 0, 0))

            scene.add(tx)
            transmitters_used.append(tx_i)

        # Run radio map solver
        rm = solver(
            scene,
            seed=seed,
            measurement_surface=surface,
            samples_per_tx=n_samples_per_tx,
            max_depth=max_depth,
        )

        # Update running maximum
        max_path_gain_i = dr.max(rm.path_gain, axis=0).numpy()

        if rm_max_path_gain is None:
            rm_max_path_gain = max_path_gain_i
        else:
            rm_max_path_gain = np.maximum(rm_max_path_gain, max_path_gain_i)

        del rm

    assert transmitters_used == list(range(n_transmitters))

    return rm_max_path_gain, tx_pos_np


def get_transmitters_for_tile(
    tile_corners: np.ndarray,
    tx_db: BaseStationDB,
) -> tuple[np.ndarray, BaseStationDB]:
    region_db = tx_db.get_region(
        tile_corners[0],
        tile_corners[1],
        search_extra_m=DEFAULT_TX_SEARCH_DISTANCE_M,
        search_radius_factor=1.0,  # DEFAULT_TX_SEARCH_RADIUS_FACTOR,
    )

    # Re-center transmitters onto the tile, assuming the scene origin corresponds to the
    # bottom-left corner of the tile.
    projection_utm_epsg_code = get_utm_epsg_code_from_gps(
        tile_corners[0, 1], tile_corners[0, 0]
    )
    to_utm = Transformer.from_crs(
        DEFAULT_LATLON_CRS, projection_utm_epsg_code, always_xy=True
    )
    tx_latlon = region_db.latlon()
    tx_utm = np.vstack(to_utm.transform(tx_latlon[:, 1], tx_latlon[:, 0])).T
    tx_utm -= to_utm.transform(tile_corners[0, 1], tile_corners[0, 0])
    return tx_utm, region_db


def compute_rm_for_tiles(
    tiles_scenes_dir: str,
    output_dir: str,
    tiles_corners_fname: str | None = None,
    transmitters: str | None = None,
    measurement_surface_id: str = DEFAULT_MEASUREMENT_MESH_NAME,
    region: list[float] | None = None,
    n_samples: int = 5e8,
    overwrite: bool = False,
    only_tiles_i: set[int] | None = None,
    measurement_z_offset: float = 1.5,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load tile corner coordinates
    if tiles_corners_fname is None:
        tiles_corners_fname = os.path.join(tiles_scenes_dir, "bboxes.npz")
    tile_corners_latlon = np.load(tiles_corners_fname)["corners"]
    numbered_tiles = dict(enumerate(tile_corners_latlon))
    if region is not None:
        south, west, north, east = region

        def bbox_intersects(bbox):
            (min_lat, min_lon), (max_lat, max_lon) = bbox
            return not (
                max_lat < south or min_lat > north or max_lon < west or min_lon > east
            )

        numbered_tiles = {i: v for i, v in numbered_tiles.items() if bbox_intersects(v)}

    tile_ids = list(numbered_tiles.keys())
    # Locate all scene tiles
    tile_scenes = ensure_scenes_ready(tiles_scenes_dir, tile_ids)

    n_tiles = len(tile_ids)
    assert len(tile_scenes) == n_tiles, (
        f"Expected scene count {len(tile_scenes)} to match tile coordinates count {n_tiles}."
        f" Loaded the tile corners from: {tiles_corners_fname}."
    )

    # Load transmitter positions
    tx_db = BaseStationDB.from_file(transmitters)

    # Before starting heavy GPU memory usage, read the amount of VRAM available
    # and estimate how many radio map entries we can fit.
    max_rm_entries_per_pass, vram_available_mib = estimate_max_rm_entries_per_pass()
    print(
        f"[i] {vram_available_mib / 1024:.1f} GiB of VRAM available,"
        f" will use a maximum of {max_rm_entries_per_pass} radio map entries per pass."
    )

    # Simulate all tiles
    t0 = time.time()
    n_computed = 0
    progress = tqdm(tile_scenes, desc="Computing radio maps")
    for tile_scene_fname in progress:
        try:
            tile_i = int(os.path.basename(os.path.dirname(tile_scene_fname)))
        except ValueError:
            print(
                f'[!] Skipping scene "{tile_scene_fname}" because its name is not a valid tile index.'
            )
            continue

        if (only_tiles_i is not None) and (tile_i not in only_tiles_i):
            continue

        output_fname = os.path.join(output_dir, f"rm_{tile_i:08d}.npz")
        if not overwrite and os.path.isfile(output_fname):
            continue

        # 1. Load scene for this tile
        scene = rt.load_scene(
            tile_scene_fname, merge_shapes_exclude_regex=measurement_surface_id
        )
        max_extents = dr.max(scene.mi_scene.bbox().extents())
        if max_extents > 1e15:
            progress.write(
                f"[!] Skipping scene {tile_scene_fname} because it seems corrupted: {max_extents}."
            )
            del scene
            gc.collect()
            continue

        # 2. Select transmitters that belong in this tile
        tile_corners = tile_corners_latlon[tile_i, ...]
        tx_utm, local_tx_db = get_transmitters_for_tile(tile_corners, tx_db)

        # 3. Compute radio map
        rm_max_path_gain, tx_pos_np = compute_rm_for_tile(
            scene,
            tx_utm,
            local_tx_db,
            measurement_surface_id,
            max_rm_entries_per_pass=max_rm_entries_per_pass,
            seed=tile_i,
            tile_i=tile_i,
            n_samples=n_samples,
            measurement_z_offset=measurement_z_offset,
            writer=progress.write,
        )

        # 4. Save results
        np.savez(
            output_fname,
            rm=rm_max_path_gain,
            tx_positions=tx_pos_np,
            measurement_z_offset=measurement_z_offset,
        )

        n_computed += 1
        gc.collect()

    elapsed = time.time() - t0
    print(
        f"[+] Finished computing {n_computed} new radio maps,"
        f" took {elapsed / max(n_computed, 1):.3f} seconds / map on average."
    )
