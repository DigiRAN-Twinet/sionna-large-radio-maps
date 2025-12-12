#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
from os.path import realpath, join, dirname, exists

if "SLRM_DATA_DIR" in os.environ:
    SLRM_DATA_DIR = os.environ["SLRM_DATA_DIR"]
else:
    SLRM_DATA_DIR = join(dirname(__file__), "..", "data")
SLRM_DATA_DIR = realpath(SLRM_DATA_DIR)

REMOTE_SCENES_DIR = realpath(join(SLRM_DATA_DIR, "remote", "scenes"))
LOCAL_SCENES_DIR = realpath(join(SLRM_DATA_DIR, "local", "scenes"))
SLRM_OPTIX_CACHE_PATH = realpath(join(SLRM_DATA_DIR, "local", "optix_cache"))

if not exists(REMOTE_SCENES_DIR):
    raise FileNotFoundError(
        f"Scenes directory not found: {REMOTE_SCENES_DIR}."
        " See README.md for instructions."
    )

RESULTS_DIR = join(SLRM_DATA_DIR, "remote", "outputs")

del realpath, join, dirname, exists
