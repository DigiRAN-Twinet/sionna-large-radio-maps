#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os

from .constants import DEFAULT_MEASUREMENT_MESH_NAME


def get_rm_results(
    tiles_scenes_dir: str, rm_results_dir: str
) -> dict[str, tuple[str, str]]:
    """
    Goes through the given directory and finds the ground mesh (measurement plane)
    for each radio map result, based on the filename.
    """
    results = {}
    for result_fname in sorted(os.listdir(rm_results_dir)):
        if not result_fname.endswith(".npz"):
            continue

        result_fname = os.path.join(rm_results_dir, result_fname)
        scene_name = os.path.basename(result_fname).strip("rm_").strip(".npz")
        ground_fname = os.path.join(
            tiles_scenes_dir, scene_name, "mesh", f"{DEFAULT_MEASUREMENT_MESH_NAME}.ply"
        )

        if not os.path.isfile(ground_fname):
            raise FileNotFoundError(
                f"Ground mesh not found for tile {scene_name}."
                f' For result file "{result_fname}", expected to find'
                ' measurement plane under "{ground_fname}".'
            )

        results[result_fname] = (ground_fname, scene_name)

    return results
