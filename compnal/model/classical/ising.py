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

from typing import Optional, Union

import numpy as np

from compnal.base_compnal import base_classical_model
from compnal.lattice import Chain, Cubic, InfiniteRange, Square
from compnal.lattice.lattice_info import LatticeType
from compnal.model.classical.model_info import ClassicalModelInfo, ClassicalModelType
from compnal.model.classical.util import determine_value_type


class Ising:
    """Class for the Ising model.

    Args:
        lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
    """

    def __init__(
        self,
        lattice: Union[Chain, Square, Cubic, InfiniteRange],
        linear: float,
        quadratic: float,
        spin_magnitude: float = 0.5,
        spin_scale_factor: int = 1,
    ) -> None:
        """Initialize Ising class.

        Args:
            lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
            linear (float): Linear interaction.
            quadratic (float): Quadratic interaction.
            spin_magnitude (float, optional): Magnitude of spins. This must be half-integer. Defaults to 0.5.
            spin_scale_factor (int, optional):
                A scaling factor used to adjust the value taken by the spin.
                The default value is 1.0, which represents the usual spin, taking value s in {-1/2, +1/2}.
                By changing this value, you can represent spins of different values,
                such as s in {-1, +1} by setting spin_scale_factor=2.
                This must be positive integer. Defaults to 1.

        Raises:
            ValueError: When the magnitude of spins or spin_scale_factor is invalid.
        """
        self._base_model = base_classical_model.make_ising(
            lattice=lattice._base_lattice,
            linear=linear,
            quadratic=quadratic,
            spin_magnitude=spin_magnitude,
            spin_scale_factor=spin_scale_factor,
        )
        self._lattice = lattice

    def get_linear(self) -> float:
        """Get the linear interaction.

        Returns:
            float: Linear interaction.
        """
        return self._base_model.get_linear()

    def get_quadratic(self) -> float:
        """Get the quadratic interaction.

        Returns:
            float: Quadratic interaction.
        """
        return self._base_model.get_quadratic()

    def get_spin_magnitude(self) -> dict[Union[list, tuple], float]:
        """Get the magnitude of spins.

        Returns:
            dict[Union[list, tuple], float]: Magnitude of spins.
                The keys are the coordinates of the lattice and the values are the magnitude of spins.
        """
        return dict(
            zip(
                self._lattice.generate_coordinate_list(),
                [v / 2 for v in self._base_model.get_twice_spin_magnitude()],
            )
        )

    def get_spin_scale_factor(self) -> int:
        """Get the spin scale factor.

        Returns:
            int: Spin scale factor.
        """
        return self._base_model.get_spin_scale_factor()

    def set_spin_magnitude(
        self, spin_magnitude: float, coordinate: Union[int, tuple]
    ) -> None:
        """Set the magnitude of spins.

        Args:
            spin_magnitude (float): Magnitude of spins. This must be half-integer.
            coordinate (Union[int, tuple]): Coordinate of the lattice.

        Raises:
            ValueError: When the magnitude of spins is invalid or the coordinate is invalid.
        """
        self._base_model.set_spin_magnitude(spin_magnitude, coordinate)

    def get_appropriate_spin_type(self) -> type:
        """Calculate appropriate spin type.

        Returns:
            type: Appropriate spin type.
        """
        scale_factor = self._base_model.get_spin_scale_factor()
        max_spin_value = 0
        for twice_spin_magnitude in self._base_model.get_twice_spin_magnitude():
            v = scale_factor * twice_spin_magnitude / 2
            if not v.is_integer():
                return np.float64
            max_spin_value = max(max_spin_value, v)
        return determine_value_type(int(max_spin_value))

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return self.export_info().to_serializable()

    @classmethod
    def from_serializable(cls, obj: dict) -> Ising:
        """Create an Ising class from a serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            Ising: Ising class.
        """
        if obj["lattice"]["lattice_type"] == LatticeType.CHAIN:
            lattice = Chain.from_serializable(obj["lattice"])
        elif obj["lattice"]["lattice_type"] == LatticeType.SQUARE:
            lattice = Square.from_serializable(obj["lattice"])
        elif obj["lattice"]["lattice_type"] == LatticeType.CUBIC:
            lattice = Cubic.from_serializable(obj["lattice"])
        elif obj["lattice"]["lattice_type"] == LatticeType.INFINITE_RANGE:
            lattice = InfiniteRange.from_serializable(obj["lattice"])
        else:
            raise ValueError("Invalid lattice type.")

        ising = cls(
            lattice=lattice,
            linear=obj["interactions"][1],
            quadratic=obj["interactions"][2],
            spin_scale_factor=obj["spin_scale_factor"],
        )

        for coordinate, spin_magnitude in zip(
            obj["spin_magnitude_keys"], obj["spin_magnitude_values"]
        ):
            ising.set_spin_magnitude(spin_magnitude, coordinate)

        return ising

    def export_info(self) -> ClassicalModelInfo:
        """Export model information.

        Returns:
            ClassicalModelInfo: Model information.
        """
        return ClassicalModelInfo(
            model_type=ClassicalModelType.ISING,
            interactions={1: self.get_linear(), 2: self.get_quadratic()},
            spin_magnitude=self.get_spin_magnitude(),
            spin_scale_factor=self.get_spin_scale_factor(),
            lattice=self._lattice.export_info(),
        )

    @property
    def lattice(self) -> Union[Chain, Square, Cubic, InfiniteRange]:
        return self._lattice
