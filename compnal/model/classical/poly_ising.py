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
        interaction: dict[int, float]
    ) -> None:
        """Initialize PolyIsing class.

        Args:
            lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
            interaction (dict[int, float]): Interaction. The key is the degree of the interaction and the value is the interaction.

        Raises:
            ValueError: When the degree of the interaction is invalid.
        """
        self._base_model = base_classical_model.make_polynomial_ising(
            lattice=lattice._base_lattice, interaction=interaction
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