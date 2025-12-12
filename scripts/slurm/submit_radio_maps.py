#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from glob import glob
import json
import os
import subprocess

from tqdm import tqdm

from common import add_project_root_to_path

add_project_root_to_path()

# pylint: disable=wrong-import-position
from scripts.compute_radio_maps import get_parser

# pylint: enable=wrong-import-position

SLURM_COMMAND = "sbatch"
SLURM_ACCOUNT = "<your_account>"
SLURM_PARTITION = "<your_partition>"
SLURM_TIME = "7:59:00"

BATCH_TEMPLATE = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gpus={gpus}
#SBATCH --time={time}
#SBATCH --exclusive

cd <path_to_repo_root>
source ./.venv/bin/activate
for ((i = 0; i < {gpus}; i++)); do
    export CUDA_VISIBLE_DEVICES=$i
    echo "--- Launching job $job_name on GPU $CUDA_VISIBLE_DEVICES"
    python ./scripts/slurm/radio_maps_worker.py {script_args} &
done

echo "Waiting for background jobs to finish..."
wait
"""


def kwargs_to_script_args(kwargs: dict) -> str:
    args_list = []
    for k, v in kwargs.items():
        if v is None:
            continue

        k = k.replace("_", "-")

        if isinstance(v, bool):
            if v:
                args_list.append(f"--{k}")
            else:
                args_list.append(f"--no-{k}")
        elif isinstance(v, str):
            args_list.append(f"--{k}='{v}'")
        else:
            args_list.append(f"--{k}={v}")
    return " ".join(args_list)


def submit_job(batch_dir: str, job_name: str, gpus: int, **kwargs):
    batch_fname = os.path.join(batch_dir, f"{job_name}.batch")
    with open(batch_fname, "w", encoding="utf-8") as f:
        script_args = kwargs_to_script_args(kwargs)
        f.write(
            BATCH_TEMPLATE.format(
                account=SLURM_ACCOUNT,
                job_name=job_name,
                partition=SLURM_PARTITION,
                gpus=gpus,
                time=SLURM_TIME,
                script_args=script_args,
            )
        )

    cmd = [SLURM_COMMAND, batch_fname]
    print("[+] Submitting job:", " ".join(cmd))
    subprocess.check_call(cmd)


def main(
    jobs: int,
    gpus: int,
    tiles_scenes_dir: str,
    output_dir: str,
    overwrite: bool,
    resume: bool,
    region: list[float] | None = None,
    **kwargs,
):
    if region is not None:
        raise NotImplementedError("--region argument is not supported yet.")

    # Figure out which results are missing from the output directory.
    # We expect one .npz for each tile.
    missing_results = []
    for scene_dir in os.listdir(tiles_scenes_dir):
        try:
            tile_i = int(scene_dir)
        except ValueError:
            continue

        output_fname = os.path.join(output_dir, f"rm_{tile_i:08d}.npz")
        if not overwrite and os.path.isfile(output_fname):
            continue
        missing_results.append(tile_i)

    print(f"[i] Found {len(missing_results)} tiles with missing results.")

    batch_dir = os.path.join(output_dir, "batch")
    pending_dir = os.path.join(output_dir, "pending")
    processing_dir = os.path.join(output_dir, "processing")
    done_dir = os.path.join(output_dir, "done")
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(pending_dir, exist_ok=True)
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    if resume:
        n_pending = len(glob(os.path.join(pending_dir, "*.json")))
        if n_pending == 0:
            print(
                f'[i] Passed --resume but no pending jobs found in "{pending_dir}", exiting.'
            )
            return
    else:
        # Create a "pending job" file for each missing result.
        for tile_i in tqdm(missing_results, desc="Creating pending job files"):
            with open(
                os.path.join(pending_dir, f"{tile_i:08d}.json"), "w", encoding="utf-8"
            ) as f:
                contents = {
                    "tile_i": tile_i,
                    "tile_scene_fname": os.path.join(
                        tiles_scenes_dir, f"{tile_i:08d}/scene.xml"
                    ),
                }
                f.write(json.dumps(contents))

    print(f"[i] Created {len(missing_results)} job descriptors in: {pending_dir}")

    for job_i in range(jobs):
        submit_job(
            batch_dir,
            f"radio_maps_{job_i:03d}",
            gpus,
            tiles_scenes_dir=tiles_scenes_dir,
            output_dir=output_dir,
            overwrite=overwrite,
            **kwargs,
        )
    print(f"[i] Submitted {jobs} jobs with {gpus} GPUs each.")


if __name__ == "__main__":
    parser = get_parser()
    parser.description = "Submit SLURM jobs to compute radio maps on the given scenes using N nodes (= jobs), M GPUs per node."

    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        help="How many jobs to schedule, i.e. how many nodes to use.",
        default=8,
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=int,
        help="How many GPUs to use on each node.",
        default=4,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Do not re-create the pending job files.",
    )

    args = parser.parse_args()
    main(**vars(args))
