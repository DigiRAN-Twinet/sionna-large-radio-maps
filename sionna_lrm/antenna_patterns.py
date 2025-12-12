#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import drjit as dr
from sionna import rt
from sionna.rt.antenna_pattern import (
    PolarizedAntennaPattern,
    register_antenna_pattern,
    v_tr38901_pattern,
)


def v_triple_tr38901_pattern(theta: float, phi: float) -> float:
    """
    Triple sector antenna pattern: simulates 3 base stations with 120° sector spacing.
    """
    offset = dr.deg2rad(120)
    v0 = v_tr38901_pattern(theta, phi)
    v1 = v_tr38901_pattern(theta, phi + offset)
    v2 = v_tr38901_pattern(theta, phi - offset)

    # Note: can't take `max` directly because they're complex numbers.
    a0 = dr.abs(v0)
    a1 = dr.abs(v1)
    a2 = dr.abs(v2)
    return dr.select(
        a0 > a1,
        dr.select(a0 > a2, v0, v2),
        dr.select(a1 > a2, v1, v2),
    )


def create_factory(name: str):
    def f(*, polarization, polarization_model="tr38901_2"):
        return PolarizedAntennaPattern(
            v_pattern=globals()["v_" + name + "_pattern"],
            polarization=polarization,
            polarization_model=polarization_model,
        )

    return f


# Register our custom antenna pattern
for _name in ["triple_tr38901"]:
    register_antenna_pattern(_name, create_factory(_name))
del _name
