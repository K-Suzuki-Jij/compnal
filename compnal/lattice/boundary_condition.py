from enum import Enum

from compnal.base_compnal import base_lattice


class BoundaryCondition(Enum):
    """Boundary condition.

    Args:
        NONE: None type.
        OBC: Open boundary condition.
        PBC: Periodic boundary condition.
    """

    NONE = 0
    OBC = 1
    PBC = 2


def cast_base_boundary_condition(boundary_condition: base_lattice.BoundaryCondition):
    """Cast base_compnal.base_lattice.BoundaryCondition to compnal.lattice.BoundaryCondition.

    Args:
        boundary_condition (base_lattice.BoundaryCondition): The boundary condition.

    Raises:
        RuntimeError: When unknown BoundaryCondition is input.

    Returns:
        BoundaryCondition: The boundary condition as the type of compnal.lattice.BoundaryCondition.
    """
    if boundary_condition == base_lattice.BoundaryCondition.NONE:
        return BoundaryCondition.NONE
    elif boundary_condition == base_lattice.BoundaryCondition.OBC:
        return BoundaryCondition.OBC
    elif boundary_condition == base_lattice.BoundaryCondition.PBC:
        return BoundaryCondition.PBC
    else:
        raise RuntimeError("Unknown BoundaryCondition")


def cast_boundary_condition(boundary_condition: BoundaryCondition):
    """Cast compnal.lattice.BoundaryCondition to base_compnal.base_lattice.BoundaryCondition.

    Args:
        boundary_condition (lattice.BoundaryCondition): The boundary condition.

    Raises:
        RuntimeError: When unknown BoundaryCondition is input.

    Returns:
        BoundaryCondition: The boundary condition as the type of base_compnal.base_lattice.BoundaryCondition.
    """
    if boundary_condition == BoundaryCondition.NONE:
        return base_lattice.BoundaryCondition.NONE
    elif boundary_condition == BoundaryCondition.OBC:
        return base_lattice.BoundaryCondition.OBC
    elif boundary_condition == BoundaryCondition.PBC:
        return base_lattice.BoundaryCondition.PBC
    else:
        raise RuntimeError("unknown BoundaryCondition")
