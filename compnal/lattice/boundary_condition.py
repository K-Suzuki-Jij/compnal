#  Copyright 2023 Kohei Suzuki
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from enum import Enum
from typing import Union

from compnal.base_compnal import base_lattice


class BoundaryCondition(str, Enum):
    """Boundary condition.

    Args:
        NONE: None type. Used for the case that the boundary condition cannot be defined.
        OBC: Open boundary condition.
        PBC: Periodic boundary condition.
    """

    NONE = "NONE"
    OBC = "OBC"
    PBC = "PBC"


def _cast_base_boundary_condition(
    base_boundary_condition: base_lattice.BoundaryCondition,
) -> BoundaryCondition:
    """Cast base boundary condition to boundary condition.

    Args:
        base_boundary_condition (base_lattice.BoundaryCondition): Boundary condition in base_compnal.

    Raises:
        ValueError: When unknown boundary condition is input.

    Returns:
        BoundaryCondition: Boundary condition in compnal.
    """

    if base_boundary_condition == base_lattice.BoundaryCondition.NONE:
        return BoundaryCondition.NONE
    elif base_boundary_condition == base_lattice.BoundaryCondition.OBC:
        return BoundaryCondition.OBC
    elif base_boundary_condition == base_lattice.BoundaryCondition.PBC:
        return BoundaryCondition.PBC
    else:
        raise ValueError("Invalid boundary condition.")


def _cast_boundary_condition(
    boundary_condition: Union[str, BoundaryCondition]
) -> base_lattice.BoundaryCondition:
    """Cast boundary condition to base boundary condition.

    Args:
        boundary_condition (Union[str, BoundaryCondition]): Boundary condition in compnal.

    Raises:
        ValueError: When unknown boundary condition is input.

    Returns:
        base_lattice.BoundaryCondition: Boundary condition in base_compnal.
    """

    if boundary_condition in (BoundaryCondition.NONE, "NONE", None):
        return base_lattice.BoundaryCondition.NONE
    elif boundary_condition in (BoundaryCondition.OBC, "OBC"):
        return base_lattice.BoundaryCondition.OBC
    elif boundary_condition in (BoundaryCondition.PBC, "PBC"):
        return base_lattice.BoundaryCondition.PBC
    else:
        raise ValueError("Invalid boundary condition.")
