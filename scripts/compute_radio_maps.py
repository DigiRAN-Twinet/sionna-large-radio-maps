#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
from multiprocessing import set_start_method

from common import add_project_root_to_path

add_project_root_to_path()

from sionna_lrm.constants import (
    DEFAULT_MEASUREMENT_MESH_NAME,
    DEFAULT_TRANSMITTERS_FNAME,
)
from sionna_lrm.radio_maps import compute_rm_for_tiles
from sionna_lrm.scene.utils import ensure_scenes_ready


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiles-scenes-dir",
        "--scenes",
        "-s",
        type=str,
        required=True,
        help="Path to the directory containing the unzipped tiles scenes.",
    )
    parser.add_argument(
        "--bboxes",
        "--tiles-corners-fname",
        "-b",
        type=str,
        default=None,
        dest="tiles_corners_fname",
        help="Path to the .npz file containing the tile corner coordinates. "
        "If not given, we will try using `bboxes.npz` under the tiles scenes directory.",
    )
    parser.add_argument(
        "--transmitters",
        "-t",
        type=str,
        help=f'Path to the transmitters file. Will use "{DEFAULT_TRANSMITTERS_FNAME}" by default.',
        default=DEFAULT_TRANSMITTERS_FNAME,
    )
    parser.add_argument(
        "--measurement-surface",
        "--measurement-surface-id",
        type=str,
        default=DEFAULT_MEASUREMENT_MESH_NAME,
        dest="measurement_surface_id",
    )
    parser.add_argument(
        "--samples", "--n-samples", type=int, default=int(5e8), dest="n_samples"
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=None,
        help="Bounding box as [south west north east] (in degrees) to which processing should be restricted.",
    )

    parser.add_argument("--output-dir", "-o", required=True, type=str)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    return parser


def main():
    parser = get_parser()
    parser.add_argument("--extract-scenes-only", action="store_true")
    args = parser.parse_args()

    if args.extract_scenes_only:
        set_start_method("spawn")
        ensure_scenes_ready(
            args.tiles_scenes_dir, progress=True, n_processes=16, allow_missing=True
        )
        return
    del args.extract_scenes_only

    return compute_rm_for_tiles(**vars(args))


if __name__ == "__main__":
    main()
