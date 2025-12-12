#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import os
import subprocess

import drjit as dr
import mitsuba as mi
import numpy as np
from sionna import rt


def sample_tx_positions(
    scene: rt.Scene,
    n_transmitters: int,
    probability_overrides: dict[str, float] | None = None,
    rng: mi.Sampler | None = None,
    z_offset: float = 0.5,
) -> np.ndarray:
    mi_scene = scene.mi_scene

    if rng is None:
        rng = mi.load_dict({"type": "independent"})
        rng.seed(1234, n_transmitters)

    # Pick meshes at random on which to spawn transmitters.
    # Assign lower weight to the terrain and no weight to the water.
    shape_ids = [sh.id() for sh in scene.mi_scene.shapes()]
    pmf = [1.0] * len(mi_scene.shapes())
    if probability_overrides is not None:
        for k, v in probability_overrides.items():
            pmf[shape_ids.index(k)] = v

    distr = mi.DiscreteDistribution(mi.Float(pmf))

    # mesh_indices = mi.UInt32(rng.next_1d() * len(mi_scene.shapes()))
    mesh_indices = distr.sample(rng.next_1d())
    meshes = dr.gather(mi.ShapePtr, mi_scene.shapes_dr(), mesh_indices)

    ps = meshes.sample_position(time=0, sample=rng.next_2d())

    # Trace rays straight down to find the highest point at the sampled locations
    si = get_highest_at_positions(mi_scene, ps.p)
    assert dr.all(si.is_valid())

    # Add transmitters at the sampled locations (with a slight vertical offset)
    si.p.z += z_offset
    return si.p.numpy()


def get_highest_at_positions(
    mi_scene: mi.Scene,
    p: mi.Point2f | mi.Point3f,
    fallback_to_scene_max: bool = False,
    allow_miss: bool = False,
):
    max_altitude = mi_scene.bbox().max.z

    rays = mi.Ray3f(o=mi.Point3f(p.x, p.y, max_altitude + 10), d=mi.Vector3f(0, 0, -1))
    si = mi_scene.ray_intersect(rays)
    if not allow_miss and dr.width(p) > 1 and not dr.any(si.is_valid()):
        raise ValueError(
            "None of the given points seem to be above the scene,"
            " perhaps there's something wrong with the coordinate system?"
        )

    if fallback_to_scene_max:
        si.p[~si.is_valid()] = mi.Point3f(p.x, p.y, max_altitude)

    return si


def split_work_into_passes(
    n_transmitters: int,
    n_measurement_faces: int,
    n_samples: int,
    max_rm_entries_per_pass: int,
    min_samples_per_tx: int,
    verbose: bool = False,
) -> tuple[int, int, int]:

    # Ensure that the sample count per transmitter hits the desired minimum.
    n_samples_per_tx = max(min_samples_per_tx, n_samples // n_transmitters)

    # Memory usage is dominated by the actual radio map output tensor, which
    # has size (n_tx, n_measurement_faces).
    # Since each transmitter's contributions are stored separately, we can split
    # up the transmitter count into multiple passes.
    n_tx_per_pass = min(
        max(1, max_rm_entries_per_pass // n_measurement_faces),
        2**32 // n_samples_per_tx,
    )
    n_passes = int(np.ceil(n_transmitters / n_tx_per_pass))

    # Now that we know we must have `n_passes`, divide the transmitters evenly
    # across the passes.
    n_tx_per_pass = int(np.ceil(n_transmitters / n_passes))

    if verbose:
        print("[i] Splitting work into passes:")
        print(f"- {n_measurement_faces=}, {n_transmitters=}, {n_samples=}.")
        print(f"- Using {n_passes} passes, {n_tx_per_pass=} and {n_samples_per_tx=}.")

    assert n_passes > 0, f"Must have at least one pass, but found {n_passes=}."
    assert n_samples_per_tx >= min_samples_per_tx, (
        f"Must have at least {min_samples_per_tx=} samples per transmitter,"
        f" but found {n_samples_per_tx=}."
    )
    assert (n_measurement_faces * n_tx_per_pass) <= max_rm_entries_per_pass, (
        f"Must have at most {max_rm_entries_per_pass=} radio map entries per pass,"
        f" but found {n_measurement_faces * n_tx_per_pass=}."
    )
    assert (n_passes * n_tx_per_pass) >= n_transmitters, (
        f"Must have at least {n_transmitters=} transmitters,"
        f" but found {n_passes * n_tx_per_pass=}."
    )

    return n_passes, n_tx_per_pass, n_samples_per_tx


def get_gpu_available_memory_mib(gpu_i: int | None = None) -> int:
    if gpu_i is None:
        gpu_i = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
        # In case CUDA_VISIBLE_DEVICES is a comma-separated list, use the first one.
        if isinstance(gpu_i, str):
            gpu_i = gpu_i.split(",")[0]

    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_i),
        ]
    )
    return int(result)
