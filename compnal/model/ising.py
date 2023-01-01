from typing import Union
from types import MappingProxyType
from numbers import Number

from compnal.base_compnal import base_model
from compnal.lattice.boundary_condition import (
    BoundaryCondition,
    cast_base_boundary_condition,
)
from compnal.lattice.one_dimension import Chain
from compnal.lattice.two_dimension import Square
from compnal.lattice.three_dimension import Cubic
from compnal.lattice.higher_dimension import AnyLattice, InfiniteRange


class Ising:
    """Ising class.

    Attributes:
        system_size (int): The system size
        boundary_condition (BoundaryCondition): The boundary condition.
        lattice (Union[Chain, Square, Cubic, InfiniteRange, AnyLattice]): The lattice on which the Ising system defined.
    """

    def __init__(
        self,
        lattice: Union[Chain, Square, Cubic, InfiniteRange, AnyLattice],
        linear: Union[float, dict[Union[int, str, tuple[Union[int, str]]], float]],
        quadratic: Union[float, dict[tuple[Union[int, str, tuple[Union[int, str]]]], float]],
    ) -> None:
        """Constructor.

        Args:
            lattice (Union[Chain, Square, Cubic, InfiniteRange, AnyLattice]): The lattice.
            linear (Union[float, dict[Union[int, str, tuple[Union[int, str]]], float]]): The linear interaction.
            quadratic (Union[float, dict[tuple[Union[int, str, tuple[Union[int, str]]]], float]]): The quadratic interaction.
        """

        self.__base_model = base_model.make_ising(
            lattice=lattice, linear=linear, quadratic=quadratic
        )

        if isinstance(linear, Number):
            self.__linear = linear
        elif isinstance(linear, dict):
            self.__linear = MappingProxyType(linear)
        else:
            raise TypeError("Invalid type for linear argument")

        if isinstance(quadratic, Number):
            self.__quadratic = linear
        elif isinstance(quadratic, dict):
            self.__quadratic = MappingProxyType(quadratic)
        else:
            raise TypeError("Invalid type for quadratic argument")

        self.__lattice = lattice

    def get_linear(self) -> Union[float, dict[Union[int, str, tuple[Union[int, str]]], float]]:
        """Get the linear interaction.

        Returns:
            Union[float, dict[Union[int, str, tuple[Union[int, str]]], float]]: The linear interaction.
        """
        return self.__linear

    def get_quadratic(self) -> Union[float, dict[tuple[Union[int, str, tuple[Union[int, str]]]], float]]:
        """Get the quadratic interaction.

        Returns:
            Union[float, dict[tuple[Union[int, str, tuple[Union[int, str]]]], float]]: The quadratic interaction.
        """
        return self.__quadratic

    def get_system_size(self) -> int:
        """Get the system size.

        Returns:
            int: The system size.
        """
        return self.__base_model.get_system_size()

    def get_boundary_condition(self) -> BoundaryCondition:
        """Get the boundary condition.

        Returns:
            BoundaryCondition: The boundary condition.
        """
        return cast_base_boundary_condition(self.__base_model.get_boundary_condition())

    def generate_index_list(self) -> list:
        """Generate the index list.

        Returns:
            list: The index list.
        """
        return self.__base_model.generate_index_list()

    def get_degree(self) -> int:
        """Get the degree of the interactions.

        Returns:
            int: The degree of the interactions.
        """
        return self.__base_model.get_degree()

    def calculate_energy(self, sample: list[int]) -> float:
        """Calculate energy.

        Args:
            sample (list[int]): Spin configuration.

        Returns:
            float: The energy.
        """
        return self.__base_model.calculate_energy(sample)

    @property
    def system_size(self) -> int:
        return self.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return self.get_boundary_condition()

    @property
    def lattice(self):
        return self.__lattice

    @property
    def _base_model(self) -> None:
        return self.__base_model