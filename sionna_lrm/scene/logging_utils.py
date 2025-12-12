#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os


class SceneNameFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.scene_name = None

    def set_scene(self, scene_name):
        self.scene_name = scene_name

    def filter(self, record):
        record.scene_name = self.scene_name or "N/A"
        return True


def setup_logging(parallel=False, scene_name=None):
    # Create a base logger (the root logger)
    pid = os.getpid()
    logger = logging.getLogger(f"p{pid}")
    log_file = f"p{pid}.log" if parallel else "debug.log"
    if not logger.hasHandlers():
        logger.setLevel(
            logging.DEBUG
        )  # The root logger level should allow all messages

        # Create a formatter for consistent output
        if scene_name is not None:
            formatter = logging.Formatter(
                "%(asctime)s - [%(scene_name)s] - [%(levelname)s] - %(message)s"
            )
        else:
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        # For console output, a simpler format:
        # e.g. "[INFO] Checking bounding box: http://bboxfinder.com/..."
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")

        # 1) Console handler at INFO level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # 2) File handler at DEBUG level
        file_handler = logging.FileHandler(log_file, mode="w")  # Overwrite each run
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add the two handlers
        if not parallel:
            # Only log to console if running sequentially
            logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    if scene_name is not None:
        if len(logger.filters) == 0:
            scene_filter = SceneNameFilter()
            logger.addFilter(scene_filter)
        logger.filters[0].set_scene(scene_name)

    return logger
