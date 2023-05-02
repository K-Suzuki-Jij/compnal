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
        quadratic: float
    ) -> None:
        """Initialize Ising class.

        Args:
            lattice (Union[Chain, Square, Cubic, InfiniteRange]): Lattice.
            linear (float): Linear interaction.
            quadratic (float): Quadratic interaction.
        """
        self._base_model = base_classical_model.make_ising(
            lattice=lattice._base_lattice, linear=linear, quadratic=quadratic
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
    
    def calculate_energy(self, state: np.array) -> float:
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