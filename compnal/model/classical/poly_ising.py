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


class PolyIsing:
    """Class for the polynomial Ising model.

    Args:
        lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
    """
    def __init__(
        self,
        lattice: Union[Chain, Square, Cubic, InfiniteRange],
        interaction: dict[int, float],
        spin_magnitude: float = 0.5,
        spin_scale_factor: int = 1
    ) -> None:
        """Initialize PolyIsing class.

        Args:
            lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
            interaction (dict[int, float]): Interaction. The key is the degree of the interaction and the value is the interaction.
            spin_magnitude (float, optional): Magnitude of spins. This must be half-integer. Defaults to 0.5.
            spin_scale_factor (int, optional): 
                A scaling factor used to adjust the value taken by the spin.
                The default value is 1.0, which represents the usual spin, taking value s\in\{-1/2,+1/2\}.
                By changing this value, you can represent spins of different values,
                such as s\in\{-1,+1\} by setting spin_scale_factor=2.
                This must be positive integer. Defaults to 1.

        Raises:
            ValueError: When the degree of the interaction is invalid or the magnitude of spins is invalid.
        """
        self._base_model = base_classical_model.make_polynomial_ising(
            lattice=lattice._base_lattice, 
            interaction=interaction, 
            spin_magnitude=spin_magnitude,
            spin_scale_factor=spin_scale_factor
        )
        self._lattice = lattice

    def get_interaction(self) -> dict[int, float]:
        """Get the interaction.

        Returns:
            dict[int, float]: Interaction.
        """
        return self._base_model.get_interaction()
    
    def get_degree(self) -> int:
        """Get the degree of the polynomial interactions.

        Returns:
            int: Degree.
        """
        return self._base_model.get_degree()
    
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
    
    def calculate_energy(self, state: Union[np.array, list]) -> float:
        """Calculate the energy of the state.

        Args:
            state (np.array): State.

        Raises:
            ValueError: When the shape of the state is invalid.

        Returns:
            float: Energy.
        """
        state = np.array(state)

        # Check if the state is valid
        if any([isinstance(self._lattice, Chain) and state.shape != (self._lattice.system_size,),
                isinstance(self._lattice, Square) and state.shape != (self._lattice.x_size, self._lattice.y_size),
                isinstance(self._lattice, Cubic) and state.shape != (self._lattice.x_size, self._lattice.y_size, self._lattice.z_size),
                isinstance(self._lattice, InfiniteRange) and state.shape != (self._lattice.system_size,)]):
            raise ValueError("The shape of the state is invalid.")
        
        return self._base_model.calculate_energy(state=state.reshape(-1))
    
    @property
    def lattice(self) -> Union[Chain, Square, Cubic, InfiniteRange]:
        return self._lattice