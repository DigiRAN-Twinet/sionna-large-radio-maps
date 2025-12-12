#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import logging
import json
import gc
import os
import random
import shutil
from tempfile import gettempdir
import time
import traceback

import numpy as np
import sionna.rt as rt

from common import add_project_root_to_path

add_project_root_to_path()

# pylint: disable=wrong-import-position
from scripts.compute_radio_maps import get_parser
from sionna_lrm.base_stations import BaseStationDB
from sionna_lrm.radio_maps import (
    compute_rm_for_tile,
    estimate_max_rm_entries_per_pass,
    get_transmitters_for_tile,
)
from sionna_lrm.scene.utils import ensure_scenes_ready

# pylint: enable=wrong-import-position

MAX_JOB_GET_RETRIES = 10000


def get_next_pending_job(
    logger: logging.Logger, pending_dir: str, processing_dir: str
) -> str | None:

    # Keep retrying because another running job may try getting the same job.
    n_attemps = 0
    while n_attemps < MAX_JOB_GET_RETRIES:
        n_attemps += 1

        available = [f for f in os.listdir(pending_dir) if f.endswith(".json")]
        if not available:
            logger.debug("No more pending jobs.")
            return None

        # Note: if using many workers, it would probably be better to try grabbing
        # a few jobs at a time so that we don't have to do this too often.
        chosen = random.choice(available)
        try:
            destination = os.path.join(processing_dir, chosen)
            shutil.move(os.path.join(pending_dir, chosen), destination)
            return destination

        except FileNotFoundError:
            logger.debug(f"Job {chosen} was likely taken by another worker, retrying.")
            continue

    logger.warning(f"Failed to get a job after {n_attemps} attempts, giving up.")
    return None


def process_job(
    logger: logging.Logger,
    job: dict,
    tile_corners_latlon: np.ndarray,
    tx_db: BaseStationDB,
    measurement_surface_id: str,
    max_rm_entries_per_pass: int,
    **kwargs,
) -> dict:
    tile_i = job["tile_i"]
    tile_scene_fname = job["tile_scene_fname"]

    # 1. Load scene for this tile
    scene = rt.load_scene(
        tile_scene_fname, merge_shapes_exclude_regex=measurement_surface_id
    )

    # 2. Select transmitters that belong in this tile
    tx_utm, local_tx_db = get_transmitters_for_tile(
        tile_corners_latlon[tile_i, ...], tx_db
    )

    # 3. Compute radio map
    rm_max_path_gain, tx_pos_np = compute_rm_for_tile(
        scene,
        tx_utm,
        local_tx_db,
        measurement_surface_id=measurement_surface_id,
        max_rm_entries_per_pass=max_rm_entries_per_pass,
        seed=tile_i,
        tile_i=tile_i,
        writer=logger.info,
        **kwargs,
    )

    # 4. Save results
    to_save = {
        "rm": rm_max_path_gain,
        "tx_positions": tx_pos_np,
    }
    if "measurement_z_offset" in kwargs:
        to_save["measurement_z_offset"] = kwargs["measurement_z_offset"]

    return to_save


def main(
    logger: logging.Logger,
    tiles_scenes_dir: str,
    output_dir: str,
    transmitters: str,
    tiles_corners_fname: str | None = None,
    region: list[float] | None = None,
    **kwargs,
):
    if region is not None:
        raise NotImplementedError("--region argument is not supported yet.")

    logger.info(
        f"Starting radio maps worker thread with:\n"
        f"- Output directory: {output_dir}\n"
        f"- PID: {os.getpid()}\n"
        f"- CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
        f"- OPTIX_CACHE_PATH: {os.environ['OPTIX_CACHE_PATH']}"
    )

    assert os.path.isdir(output_dir)

    pending_dir = os.path.join(output_dir, "pending")
    processing_dir = os.path.join(output_dir, "processing")
    done_dir = os.path.join(output_dir, "done")
    failed_dir = os.path.join(output_dir, "failed")
    os.makedirs(done_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # Locate all scene tiles
    tile_scenes = ensure_scenes_ready(tiles_scenes_dir)

    # Load tile corner coordinates
    if tiles_corners_fname is None:
        tiles_corners_fname = os.path.join(tiles_scenes_dir, "bboxes.npz")
    tile_corners_latlon = np.load(tiles_corners_fname)["corners"]
    n_tiles = tile_corners_latlon.shape[0]
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
        f"[i] {vram_available_mib/1024:.1f} GiB of VRAM available, will use a maximum of {max_rm_entries_per_pass} radio map entries per pass."
    )

    while True:
        job_fname = get_next_pending_job(logger, pending_dir, processing_dir)
        if job_fname is None:
            logger.info("No more pending jobs, exiting.")
            break

        with open(job_fname, "r", encoding="utf-8") as f:
            job = json.load(f)

        output_fname = os.path.join(output_dir, f"rm_{job['tile_i']:08d}.npz")

        try:
            t0 = time.time()
            results = process_job(
                logger,
                job,
                tile_corners_latlon,
                tx_db,
                max_rm_entries_per_pass=max_rm_entries_per_pass,
                **kwargs,
            )
            job["elapsed"] = time.time() - t0
            job["output_fname"] = output_fname

            # Save the actual results
            np.savez(output_fname, **results)

            # Add job metadata to the 'done' directory (clear error fields if present)
            fname = os.path.join(done_dir, os.path.basename(job_fname))
            job.pop("error", None)
            job.pop("traceback", None)
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(job, f)

            logger.info(f"Processed job {job_fname} in {job['elapsed']:.3f} seconds.")
            gc.collect()

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Add job metadata to the 'failed' directory
            fname = os.path.join(failed_dir, os.path.basename(job_fname))
            job["error"] = str(e)
            job["traceback"] = str(traceback.format_exc())
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(job, f)

            logger.error(f"Error processing job {job_fname}: {e}")
            logger.error(traceback.format_exc())

        finally:
            # Whether successful or not, the job is no longer pending.
            logger.debug(
                f"Removing {job_fname} job file from pending directory: {job_fname}"
            )
            os.unlink(job_fname)


if __name__ == "__main__":
    # See https://github.com/mitsuba-renderer/mitsuba3/issues/892#issuecomment-1918652252
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Try and use a unique cache directory for each GPU.
    os.environ["OPTIX_CACHE_PATH"] = os.path.join(gettempdir(), f"optix_cache_{gpu}")

    # Configure logging
    gpu_str = gpu if (gpu is not None) else "N/A"
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s, %(process)d, GPU{gpu_str}] %(levelname)s: %(message)s",
    )

    parser = get_parser()
    args = parser.parse_args()

    # At this point, some flags were already accounted for when creating the job files.
    del args.overwrite

    main(logging.getLogger(__name__), **vars(args))
