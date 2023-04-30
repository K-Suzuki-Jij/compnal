from enum import Enum
from typing import Union
from compnal.base_compnal import base_lattice

class BoundaryCondition(Enum):
    """Boundary condition.

    Args:
        NONE: None type. Used for the case that the boundary condition cannot be defined.
        OBC: Open boundary condition.
        PBC: Periodic boundary condition.
    """

    NONE = 0
    OBC = 1
    PBC = 2

def _cast_base_boundary_condition(base_boundary_condition: base_lattice.BoundaryCondition) -> BoundaryCondition:
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
    
def _cast_boundary_condition(boundary_condition: Union[str, BoundaryCondition]) -> base_lattice.BoundaryCondition:
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

    

