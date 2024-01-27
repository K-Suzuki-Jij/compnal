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

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from compnal.lattice.lattice_info import LatticeInfo


class ClassicalModelType(str, Enum):
    """Classical model type.

    Args:
        ISING: Ising model.
        POLY_ISING: Polynomial Ising model.
    """

    ISING = "ISING"
    POLY_ISING = "POLY_ISING"


@dataclass
class ClassicalModelInfo:
    """Classical model information.

    Args:
        model_type (ClassicalModelType): Model type. Defaults to None.
        interactions (dict[int, int]): Interactions. Defaults to None.
    """

    model_type: Optional[ClassicalModelType] = None
    interactions: Optional[dict[int, float]] = None
    spin_magnitude: Optional[dict[Union[list, tuple], float]] = None
    spin_scale_factor: Optional[int] = None
    lattice: Optional[LatticeInfo] = None

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return {
            "model_type": self.model_type,
            "interactions": self.interactions,
            "spin_magnitude_values": list(self.spin_magnitude.values()),
            "spin_magnitude_keys": list(self.spin_magnitude.keys()),
            "spin_scale_factor": self.spin_scale_factor,
            "lattice": self.lattice.to_serializable(),
        }

    @classmethod
    def from_serializable(cls, obj: dict):
        """Convert from a serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            ClassicalModelInfo: Classical model information.
        """
        return cls(
            model_type=obj.get("model_type", None),
            interactions=obj.get("interactions", None),
            spin_magnitude=dict(
                zip(
                    [tuple(elem) for elem in obj.get("spin_magnitude_keys", None)],
                    obj.get("spin_magnitude_values", None),
                )
            ),
            spin_scale_factor=obj.get("spin_scale_factor", None),
            lattice=LatticeInfo.from_serializable(obj.get("lattice", None)),
        )
