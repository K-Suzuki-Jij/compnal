#  Copyright 2024 Kohei Suzuki
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


from __future__ import annotations

from typing import Union

from compnal.base_compnal import base_lattice
from compnal.lattice.boundary_condition import (
    BoundaryCondition,
    _cast_base_boundary_condition,
    _cast_boundary_condition,
)
from compnal.lattice.lattice_info import LatticeInfo, LatticeType


class Cubic:
    """The class to represent the three-dimensional cubic lattice."""

    def __init__(
        self,
        x_size: int,
        y_size: int,
        z_size: int,
        boundary_condition: Union[str, BoundaryCondition],
    ) -> None:
        """Initialize cubic lattice class.

        Args:
            x_size (int): The length of the x-direction. This must be larger than zero.
            y_size (int): The length of the y-direction. This must be larger than zero.
            z_size (int): The length of the z-direction. This must be larger than zero.
            boundary_condition (Union[str, BoundaryCondition]): Boundary condition.

        Raises:
            ValueError: When x/y/z size is smaller than or equal to zero.
        """
        self._base_cubic = base_lattice.Cubic(
            x_size=x_size,
            y_size=y_size,
            z_size=z_size,
            boundary_condition=_cast_boundary_condition(boundary_condition),
        )

    def generate_coordinate_list(self) -> list[tuple[int, int, int]]:
        """Generate coordinate list.

        Returns:
            list[tuple[int, int, int]]: Coordinate list.
        """
        return self._base_cubic.generate_coordinate_list()

    def to_serializable(self) -> dict:
        """Convert to serializable object.

        Returns:
            dict: Serializable object.
        """
        return self.export_info().to_serializable()

    @classmethod
    def from_serializable(cls, obj: dict) -> Cubic:
        """Create cubic lattice from serializable object.

        Args:
            obj (dict): Serializable object.

        Raises:
            ValueError: When the lattice type is not cubic.

        Returns:
            Cubic: Cubic lattice.
        """
        if obj["lattice_type"] != LatticeType.CUBIC:
            raise ValueError("The lattice type is not cubic.")

        return cls(
            x_size=obj["shape"][0],
            y_size=obj["shape"][1],
            z_size=obj["shape"][2],
            boundary_condition=obj["boundary_condition"],
        )

    def export_info(self) -> LatticeInfo:
        """Export lattice information.

        Returns:
            LatticeInfo: Lattice information.
        """
        return LatticeInfo(
            lattice_type=LatticeType.CUBIC,
            system_size=self.system_size,
            shape=(self.x_size, self.y_size, self.z_size),
            boundary_condition=self.boundary_condition,
        )

    @property
    def x_size(self) -> int:
        return self._base_cubic.get_x_size()

    @property
    def y_size(self) -> int:
        return self._base_cubic.get_y_size()

    @property
    def z_size(self) -> int:
        return self._base_cubic.get_z_size()

    @property
    def system_size(self) -> int:
        return self._base_cubic.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return _cast_base_boundary_condition(self._base_cubic.get_boundary_condition())

    @property
    def _base_lattice(self) -> base_lattice.Cubic:
        return self._base_cubic
