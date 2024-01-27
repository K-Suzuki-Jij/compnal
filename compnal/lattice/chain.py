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


class Chain:
    """The class to represent the one-dimensional chain.

    Attributes:
        system_size (int): System size.
        boundary_condition (BoundaryCondition): Boundary condition.
    """

    def __init__(
        self, system_size: int, boundary_condition: Union[str, BoundaryCondition]
    ) -> None:
        """Initialize chain class.

        Args:
            system_size (int): System size. This must be larger than zero.
            boundary_condition (Union[str, BoundaryCondition]): Boundary condition.

        Raises:
            ValueError: When the system size is smaller than or equal to zero.
        """
        self._base_chain = base_lattice.Chain(
            system_size=system_size,
            boundary_condition=_cast_boundary_condition(boundary_condition),
        )

    def generate_coordinate_list(self) -> list[tuple[int]]:
        """Generate coordinate list.

        Returns:
            list[tuple[int]]: Coordinate list.
        """
        return self._base_chain.generate_coordinate_list()

    def to_serializable(self) -> dict:
        """Convert to serializable object.

        Returns:
            dict: Serializable object.
        """
        return self.export_info().to_serializable()

    @classmethod
    def from_serializable(cls, obj: dict) -> Chain:
        """Create chain class from serializable object.

        Args:
            obj (dict): Serializable object.

        Raises:
            ValueError: When the lattice type is not chain.

        Returns:
            Chain: Chain.
        """
        if obj["lattice_type"] != LatticeType.CHAIN:
            raise ValueError("The lattice type is not chain.")

        return cls(
            system_size=obj["system_size"], boundary_condition=obj["boundary_condition"]
        )

    def export_info(self) -> LatticeInfo:
        """Export lattice information.

        Returns:
            LatticeInfo: Lattice information.
        """
        return LatticeInfo(
            lattice_type=LatticeType.CHAIN,
            system_size=self.system_size,
            shape=(self.system_size,),
            boundary_condition=self.boundary_condition,
        )

    @property
    def system_size(self) -> int:
        return self._base_chain.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return _cast_base_boundary_condition(self._base_chain.get_boundary_condition())

    @property
    def _base_lattice(self) -> base_lattice.Chain:
        return self._base_chain
