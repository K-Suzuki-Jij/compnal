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

from compnal.base_compnal import base_lattice
from compnal.lattice.boundary_condition import (
    BoundaryCondition,
    _cast_base_boundary_condition,
)
from compnal.lattice.lattice_info import LatticeInfo, LatticeType


class InfiniteRange:
    """The class to represent the infinite range lattice.

    Attributes:
        system_size (int): System size.
        boundary_condition (BoundaryCondition): Boundary condition. This is always BoundaryCondition.NONE.
    """

    def __init__(self, system_size: int) -> None:
        """Initialize infinite range lattice class.

        Args:
            system_size (int): System size. This must be larger than zero.

        Raises:
            ValueError: When the system size is smaller than or equal to zero.
        """
        self._base_infinite_range = base_lattice.InfiniteRange(system_size=system_size)

    def generate_coordinate_list(self) -> list[tuple[int]]:
        """Generate coordinate list.

        Returns:
            list[tuple[int]]: Coordinate list.
        """
        return self._base_infinite_range.generate_coordinate_list()

    def to_serializable(self) -> dict:
        """Convert to serializable object.

        Returns:
            dict: Serializable object.
        """
        return self.export_info().to_serializable()

    @classmethod
    def from_serializable(cls, obj: dict) -> InfiniteRange:
        """Generate infinite range lattice from serializable object.

        Args:
            obj (dict): Serializable object.

        Raises:
            ValueError: When the lattice type is not infinite range.

        Returns:
            InfiniteRange: Infinite range lattice.
        """
        if obj["lattice_type"] != LatticeType.INFINITE_RANGE:
            raise ValueError("The lattice type is not InfiniteRange.")

        return cls(system_size=obj["system_size"])

    def export_info(self) -> LatticeInfo:
        """Export lattice information.

        Returns:
            LatticeInfo: Lattice information.
        """
        return LatticeInfo(
            lattice_type=LatticeType.INFINITE_RANGE,
            system_size=self.system_size,
            shape=None,
            boundary_condition=self.boundary_condition,
        )

    @property
    def system_size(self) -> int:
        return self._base_infinite_range.get_system_size()

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return _cast_base_boundary_condition(
            self._base_infinite_range.get_boundary_condition()
        )

    @property
    def _base_lattice(self) -> base_lattice.InfiniteRange:
        return self._base_infinite_range
