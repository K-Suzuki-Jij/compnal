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
from typing import Union
from compnal.lattice import Chain, Square, Cubic, InfiniteRange
from compnal.base_compnal import base_classical_model
import numpy as np

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
        spin_scale_factor: int = 1
    ) -> None:
        """Initialize Ising class.

        Args:
            lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
            linear (float): Linear interaction.
            quadratic (float): Quadratic interaction.
            spin_magnitude (float, optional): Magnitude of spins. This must be half-integer. Defaults to 0.5.
            spin_scale_factor (int, optional): 
                A scaling factor used to adjust the value taken by the spin.
                The default value is 1.0, which represents the usual spin, taking value s\in\{-1/2,+1/2\}.
                By changing this value, you can represent spins of different values,
                such as s\in\{-1,+1\} by setting spin_scale_factor=2.
                This must be positive integer. Defaults to 1.

        Raises:
            ValueError: When the magnitude of spins or spin_scale_factor is invalid.
        """
        self._base_model = base_classical_model.make_ising(
            lattice=lattice._base_lattice, 
            linear=linear, 
            quadratic=quadratic, 
            spin_magnitude=spin_magnitude,
            spin_scale_factor=spin_scale_factor
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
            dict[Union[list, tuple], float]: Magnitude of spins. The keys are the coordinates of the lattice and the values are the magnitude of spins.
        """
        return dict(zip(self._lattice.generate_coordinate_list(), [v/2 for v in self._base_model.get_twice_spin_magnitude()]))
    
    def get_spin_scale_factor(self) -> int:
        """Get the spin scale factor.

        Returns:
            int: Spin scale factor.
        """
        return self._base_model.get_spin_scale_factor()

    def set_spin_magnitude(self, spin_magnitude: float, coordinate: Union[int, tuple]) -> None:
        """Set the magnitude of spins.

        Args:
            spin_magnitude (float): Magnitude of spins. This must be half-integer.
            coordinate (Union[int, tuple]): Coordinate of the lattice.
        
        Raises:
            ValueError: When the magnitude of spins is invalid or the coordinate is invalid.
        """
        self._base_model.set_spin_magnitude(spin_magnitude, coordinate)
    
    @property
    def lattice(self) -> Union[Chain, Square, Cubic, InfiniteRange]:
        return self._lattice