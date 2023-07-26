#  Copyright 2023 Kohei Suzuki
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

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from compnal.lattice.boundary_condition import BoundaryCondition


class LatticeType(str, Enum):
    """Lattice type.

    Args:
        CHAIN: Chain lattice.
        SQUARE: Square lattice.
        CUBIC: Cubic lattice.
        INFINITE_RANGE: Infinite-range lattice.
    """

    CHAIN = "CHAIN"
    SQUARE = "SQUARE"
    CUBIC = "CUBIC"
    INFINITE_RANGE = "INFINITE_RANGE"


@dataclass
class LatticeInfo:
    """Lattice information.

    Args:
        lattice_type (LatticeType): Lattice type. Defaults to None.
        system_size (int): System size. Defaults to None.
        shape (tuple[int]): Shape. Defaults to None.
        boundary_condition (BoundaryCondition): Boundary condition. Defaults to None.
    """

    lattice_type: Optional[LatticeType] = None
    system_size: Optional[int] = None
    shape: Optional[tuple[int]] = None
    boundary_condition: Optional[BoundaryCondition] = None

    def to_serializable(self) -> dict:
        """Convert to serializable object.

        Returns:
            dict: Serializable object.
        """
        return asdict(self)

    @classmethod
    def from_serializable(cls, obj: dict):
        """Convert from serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            LatticeInfo: Lattice information.
        """
        return cls(**obj)
