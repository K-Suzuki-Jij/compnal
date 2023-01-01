from compnal.base_compnal import base_lattice
from compnal.lattice.boundary_condition import (
    BoundaryCondition,
    cast_base_boundary_condition,
    cast_boundary_condition,
)


class Chain(base_lattice.Chain):
    """One-dimensional Chain lattice.

    Attributes:
        system_size (int): The system size.
        boundary_condition (BoundaryCondition): The boundary condition.
    """

    def __init__(
        self,
        system_size: int,
        boundary_condition: BoundaryCondition = BoundaryCondition.OBC,
    ) -> None:
        """
        Args:
            system_size (int): The system size.
            boundary_condition (BoundaryCondition, optional): The boundary condition. Defaults to BoundaryCondition.OBC.
        """
        super().__init__(
            system_size=system_size,
            boundary_condition=cast_boundary_condition(boundary_condition),
        )

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

    def generate_x_coordinate(self) -> list:
        """Generate the x-coordinate of the samples.

        Returns:
            list: the x-coordinate of the samples.
        """
        return [i for i in range(self.x_size)]

    def generate_coordinate(self) -> list:
        """Generate the coordinate of the samples.

        Returns:
            list: The coordinate of the samples.
        """
        return [i for i in range(self.x_size)]

    @property
    def system_size(self) -> int:
        return self.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return self.get_boundary_condition()
