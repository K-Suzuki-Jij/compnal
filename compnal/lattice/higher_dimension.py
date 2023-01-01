from compnal.base_compnal import base_lattice
from compnal.lattice.boundary_condition import (
    BoundaryCondition,
    cast_base_boundary_condition,
)


class InfiniteRange(base_lattice.InfiniteRange):
    """Infinite-range lattice.

    Args:
        system_size (int): The system size.
        boundary_condition (BoundaryCondition): The boundary condition.
    """

    def __init__(self, system_size: int) -> None:
        """
        Args:
            system_size (int): The system size.
        """
        super().__init__(system_size=system_size)

    def get_system_size(self) -> int:
        """Get the system size.

        Returns:
            int: The system size.
        """
        return super().get_system_size()

    def get_boundary_condition(self) -> BoundaryCondition:
        """Get the boundary condition.

        Returns:
            BoundaryCondition: The boundary condition.
        """
        return cast_base_boundary_condition(super().get_boundary_condition())

    def generate_index_list(self) -> list:
        """Generate the index list of the samples.

        Returns:
            list: The index list of the samples.
        """
        return [i for i in range(self.system_size)]

    @property
    def system_size(self) -> int:
        return self.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return cast_base_boundary_condition(super().get_boundary_condition())


class AnyLattice(base_lattice.AnyLattice):
    """Any types of lattice.

    Args:
        boundary_condition (BoundaryCondition): The boundary condition.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_boundary_condition(self) -> BoundaryCondition:
        """Get the boundary condition.

        Returns:
            BoundaryCondition: The boundary condition.
        """
        return cast_base_boundary_condition(super().get_boundary_condition())

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return cast_base_boundary_condition(super().get_boundary_condition())
