from __future__ import annotations
from typing import Union
from compnal.base_compnal import base_lattice
from compnal.lattice.boundary_condition import (
    BoundaryCondition, 
    _cast_base_boundary_condition, 
    _cast_boundary_condition
)


class Chain:
    """The class to represent the one-dimensional chain.

    Attributes:
        system_size (int): System size.
        boundary_condition (BoundaryCondition): Boundary condition.
    """

    def __init__(self, system_size: int, boundary_condition: Union[str, BoundaryCondition]) -> None:
        """Initialize chain class.

        Args:
            system_size (int): System size. This must be larger than zero.
            boundary_condition (Union[str, BoundaryCondition]): Boundary condition.

        Raises:
            ValueError: When the system size is smaller than or equal to zero.
        """

        if system_size <= 0:
            raise ValueError("System size must be larger than zero.")

        self._base_chain = base_lattice.Chain(
            system_size=system_size, 
            boundary_condition=_cast_boundary_condition(boundary_condition)
        )
    
    def generate_coordinate_list(self) -> list[int]:
        """Generate coordinate list.

        Returns:
            list[int]: Coordinate list.
        """
        return self._base_chain.generate_coordinate_list()
    
    @property
    def system_size(self) -> int:
        """System size.

        Returns:
            int: System size.
        """
        return self._base_chain.get_system_size()
    
    @property
    def boundary_condition(self) -> BoundaryCondition:
        """Boundary condition.

        Returns:
            BoundaryCondition: Boundary condition.
        """
        return _cast_base_boundary_condition(self._base_chain.get_boundary_condition())