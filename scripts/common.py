#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..")
    if lib_path not in sys.path:
        sys.path.append(lib_path)
