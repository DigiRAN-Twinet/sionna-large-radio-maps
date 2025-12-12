#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import os
from glob import glob
import time

from tqdm import tqdm


def get_file_count(d: str) -> int:
    return len(glob(os.path.join(d, "*.json")))


def main(output_dir: str, delay: int):
    checks = {
        "Pending": os.path.join(output_dir, "pending"),
        "Processing": os.path.join(output_dir, "processing"),
        "Done": os.path.join(output_dir, "done"),
        "Failed": os.path.join(output_dir, "failed"),
    }

    n_files = {check: get_file_count(d) for check, d in checks.items()}
    total = sum(n_files.values())
    initial = total - n_files["Pending"]
    progress = tqdm(desc="Jobs progress", total=total, initial=initial)
    while True:
        n_files = {check: get_file_count(d) for check, d in checks.items()}

        progress.update((total - n_files["Pending"]) - progress.n)
        progress.set_postfix(
            {
                "Pending": n_files["Pending"],
                "Processing": n_files["Processing"],
                "Done": n_files["Done"],
                "Failed": n_files["Failed"],
            }
        )

        if n_files["Pending"] + n_files["Processing"] == 0:
            progress.write("No more pending or processing jobs.")
            break

        time.sleep(delay)

    progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=str, help="Path to the output directory to watch."
    )
    parser.add_argument(
        "--delay", "-n", type=int, default=10, help="Delay in seconds between checks."
    )
    args = parser.parse_args()
    main(**vars(args))
