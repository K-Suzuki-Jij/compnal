from compnal.base_compnal import base_lattice
from compnal.lattice.boundary_condition import (
    BoundaryCondition,
    cast_base_boundary_condition,
    cast_boundary_condition,
)


class Cubic(base_lattice.Cubic):
    """Three-dimensional cubic lattice.

    Attributes:
        x_size (int): The size of the x-direction.
        y_size (int): The size of the y-direction.
        z_size (int): The size of the z-direction.
        system_size (int): The system size (x_size*y_size*z_size).
        boundary_condition (BoundaryCondition): The boundary condition.
    """

    def __init__(
        self,
        x_size: int,
        y_size: int,
        z_size: int,
        boundary_condition: BoundaryCondition = BoundaryCondition.OBC,
    ) -> None:
        """
        Args:
            x_size (int): The size of the x-direction.
            y_size (int): The size of the y-direction.
            z_size (int): The size of the z-direction.
            boundary_condition (BoundaryCondition, optional): The boundary condition. Defaults to BoundaryCondition.OBC.
        """
        super().__init__(
            x_size=x_size,
            y_size=y_size,
            z_size=z_size,
            boundary_condition=cast_boundary_condition(boundary_condition),
        )

    def get_x_size(self) -> int:
        """Get the size of the x-direction.

        Returns:
            int: The size of the x-direction.
        """
        return super().get_x_size()

    def get_y_size(self) -> int:
        """Get the size of the y-direction.

        Returns:
            int: The size of the y-direction.
        """
        return super().get_y_size()

    def get_z_size(self) -> int:
        """Get the size of the z-direction.

        Returns:
            int: The size of the z-direction.
        """
        return super().get_z_size()

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

    def generate_x_coordinate(self) -> list[int]:
        """Generate the x-coordinate of the samples.

        Returns:
            list: the x-coordinate of the samples.
        """
        return [i for _ in range(self.z_size) for _ in range(self.y_size) for i in range(self.x_size)]

    def generate_y_coordinate(self) -> list[int]:
        """Generate the y-coordinate of the samples.

        Returns:
            list: the y-coordinate of the samples.
        """
        return [i for _ in range(self.z_size) for i in range(self.y_size) for _ in range(self.x_size)]
    
    def generate_z_coordinate(self) -> list[int]:
        """Generate the z-coordinate of the samples.

        Returns:
            list: the z-coordinate of the samples.
        """
        return [i for i in range(self.z_size) for _ in range(self.y_size) for _ in range(self.x_size)]

    def generate_coordinate(self) -> list[tuple]:
        """Generate the coordinate of the samples.

        Returns:
            list[tuple]: The coordinate of the samples.
        """
        return [(i, j, k) for k in range(self.z_size) for j in range(self.y_size) for i in range(self.x_size)]

    @property
    def x_size(self) -> int:
        return self.get_x_size()

    @property
    def y_size(self) -> int:
        return self.get_y_size()

    @property
    def z_size(self) -> int:
        return self.get_z_size()

    @property
    def system_size(self) -> int:
        return self.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return self.get_boundary_condition()
