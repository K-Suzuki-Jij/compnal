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
from typing import Union, Optional
from compnal.model import Ising, PolyIsing
from compnal.base_solver import base_solver
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod,
    _cast_base_state_update_method,
    _cast_base_random_number_engine,
    _cast_base_spin_selection_method,
    _cast_state_update_method,
    _cast_random_number_engine,
    _cast_spin_selection_method
)
import numpy as np



class ClassicalMonteCarlo:
    """Class for the classical Monte Carlo solver.
    """
    def __init__(self, model: Union[Ising, PolyIsing]) -> None:
        """Initialize ClassicalMonteCarlo class.

        Args:
            model (Union[Ising, PolyIsing]): Model.
        """
        self._base_solver = base_solver.make_classical_monte_carlo(model._base_model)
        self._model = model

    def set_num_sweeps(self, num_sweeps: int) -> None:
        """Set the number of sweeps.

        Args:
            num_sweeps (int): Number of sweeps.

        Raises:
            ValueError: When num_sweeps < 0.
        """
        self._base_solver.set_num_sweeps(num_sweeps)

    def set_num_samples(self, num_samples: int) -> None:
        """Set the number of samples.

        Args:
            num_samples (int): Number of samples.

        Raises:
            ValueError: When num_samples <= 0.
        """
        self._base_solver.set_num_samples(num_samples)

    def set_num_threads(self, num_threads: int) -> None:
        """Set the number of threads.

        Args:
            num_threads (int): Number of threads.

        Raises:
            ValueError: When num_threads <= 0.
        """
        self._base_solver.set_num_threads(num_threads)

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature.

        Args:
            temperature (float): Temperature.
        
        Raises:
            ValueError: When temperature < 0.0.
        """
        self._base_solver.set_temperature(temperature)

    def set_state_update_method(self, state_update_method: Union[str, StateUpdateMethod]) -> None:
        """Set the state update method.

        Args:
            state_update_method (Union[str, StateUpdateMethod]): State update method. "METROPOLIS" or "HEAT_BATH".

        Raises:
            ValueError: When state_update_method is invalid.
        """
        self._base_solver.set_state_update_method(
            _cast_state_update_method(state_update_method)
        )

    def set_random_number_engine(self, random_number_engine: Union[str, RandomNumberEngine]) -> None:
        """Set the random number engine.

        Args:
            random_number_engine (Union[str, RandomNumberEngine]): Random number engine. "MT", "MT_64", or "XORSHIFT".

        Raises:
            ValueError: When random_number_engine is invalid.
        """
        self._base_solver.set_random_number_engine(
            _cast_random_number_engine(random_number_engine)
        )

    def set_spin_selection_method(self, spin_selection_method: Union[str, SpinSelectionMethod]) -> None:
        """Set the spin selection method.

        Args:
            spin_selection_method (Union[str, SpinSelectionMethod]): Spin selection method. "RANDOM" or "SEQUENTIAL".

        Raises:
            ValueError: When spin_selection_method is invalid.
        """
        self._base_solver.set_spin_selection_method(
            _cast_spin_selection_method(spin_selection_method)
        )

    def get_num_sweeps(self) -> int:
        """Get the number of sweeps.

        Returns:
            int: Number of sweeps.
        """
        return self._base_solver.get_num_sweeps()
    
    def get_num_samples(self) -> int:
        """Get the number of samples.

        Returns:
            int: Number of samples.
        """
        return self._base_solver.get_num_samples()
    
    def get_num_threads(self) -> int:
        """Get the number of threads.

        Returns:
            int: Number of threads.
        """
        return self._base_solver.get_num_threads()
    
    def get_temperature(self) -> float:
        """Get the temperature.

        Returns:
            float: Temperature.
        """
        return self._base_solver.get_temperature()
    
    def get_state_update_method(self) -> StateUpdateMethod:
        """Get the state update method.

        Returns:
            StateUpdateMethod: State update method.
        """
        return _cast_base_state_update_method(self._base_solver.get_state_update_method())
    
    def get_random_number_engine(self) -> RandomNumberEngine:
        """Get the random number engine.

        Returns:
            RandomNumberEngine: Random number engine.
        """
        return _cast_base_random_number_engine(self._base_solver.get_random_number_engine())
    
    def get_spin_selection_method(self) -> SpinSelectionMethod:
        """Get the spin selection method.

        Returns:
            SpinSelectionMethod: Spin selection method.
        """
        return _cast_base_spin_selection_method(self._base_solver.get_spin_selection_method())

    def get_seed(self) -> int:
        """Get the seed.

        Returns:
            int: Seed.
        """
        return self._base_solver.get_seed()
    
    def run_sampling(self, seed: Optional[int] = None) -> np.ndarray:
        """Run sampling.

        Args:
            seed (Optional[int], optional): Seed. Defaults to None.

        Returns:
            np.ndarray: Samples.
        """
        if seed is not None:
            return np.array(self._base_solver.run_sampling(seed))
        else:
            return np.array(self._base_solver.run_sampling())