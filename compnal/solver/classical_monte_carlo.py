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
from compnal.model.classical import Ising, PolyIsing
from compnal.lattice import Chain, Square, Cubic, InfiniteRange
from compnal.base_compnal import base_solver
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

    Attributes:
        num_sweeps (int): Number of sweeps.
        num_samples (int): Number of samples.
        num_threads (int): Number of threads.
        temperature (float): Temperature.
        state_update_method (StateUpdateMethod): State update method.
        random_number_engine (RandomNumberEngine): Random number engine.
        spin_selection_method (SpinSelectionMethod): Spin selection method.
    """
    def __init__(self, model: Union[Ising, PolyIsing]) -> None:
        """Initialize the classical Monte Carlo solver.

        Args:
            model (Union[Ising, PolyIsing]): Model.
        """
        self._base_solver = base_solver.make_classical_monte_carlo(model._base_model)
        self._model = model

    def get_seed(self) -> int:
        """Get the seed.

        Returns:
            int: Seed.
        """
        return self._base_solver.get_seed()
    
    def run_sampling(
        self,
        temperature: float,     
        num_sweeps: int = 1000,
        num_samples: int = 1,
        num_threads: int = 1,
        state_update_method: Union[str, StateUpdateMethod] = "METROPOLIS",
        random_number_engine: Union[str, RandomNumberEngine] = "MT",
        spin_selection_method: Union[str, SpinSelectionMethod] = "RANDOM",
        seed: Optional[int] = None
    ) -> np.ndarray:
        
        """Run sampling.

        Args:
            temperature (float): Temperature.
            num_sweeps (int, optional): Number of sweeps. Defaults to 1000.
            num_samples (int, optional): Number of samples. Defaults to 1.
            num_threads (int, optional): Number of threads. Defaults to 1.
            state_update_method (Union[str, StateUpdateMethod], optional): State update method. Defaults to "METROPOLIS".
            random_number_engine (Union[str, RandomNumberEngine], optional): Random number engine. Defaults to "MT".
            spin_selection_method (Union[str, SpinSelectionMethod], optional): Spin selection method. Defaults to "RANDOM".
            seed (Optional[int], optional): Seed. Defaults to None.
            
        Returns:
            np.ndarray: Samples.
        """

        self._base_solver.set_temperature(temperature)
        self._base_solver.set_num_sweeps(num_sweeps)
        self._base_solver.set_num_samples(num_samples)
        self._base_solver.set_num_threads(num_threads)
        self._base_solver.set_state_update_method(
            _cast_state_update_method(state_update_method)
        )
        self._base_solver.set_random_number_engine(
            _cast_random_number_engine(random_number_engine)
        )
        self._base_solver.set_spin_selection_method(
            _cast_spin_selection_method(spin_selection_method)
        )
        if seed is not None:
            sample_list = self._base_solver.run_sampling(seed)
        else:
            sample_list = self._base_solver.run_sampling()

        if isinstance(self._model.lattice, Chain):
            return np.array(sample_list)
        elif isinstance(self._model.lattice, Square):
            return np.reshape(
                sample_list, (self.num_samples, self._model.lattice.x_size, self._model.lattice.y_size)
            )
        elif isinstance(self._model.lattice, Cubic):
            return np.reshape(
                sample_list, 
                (self.num_samples, self._model.lattice.x_size, self._model.lattice.y_size, self._model.lattice.z_size)
            )
        elif isinstance(self._model.lattice, InfiniteRange):
            return np.array(sample_list)
        else:
            raise NotImplementedError("Not implemented.")
        
    @property
    def num_sweeps(self) -> int:
        return self._base_solver.get_num_sweeps()
    
    @num_sweeps.setter
    def num_sweeps(self, num_sweeps: int) -> None:
        self._base_solver.set_num_sweeps(num_sweeps)

    @property
    def num_samples(self) -> int:
        return self._base_solver.get_num_samples()
    
    @num_samples.setter
    def num_samples(self, num_samples: int) -> None:
        self._base_solver.set_num_samples(num_samples)

    @property
    def num_threads(self) -> int:
        return self._base_solver.get_num_threads()
    
    @num_threads.setter
    def num_threads(self, num_threads: int) -> None:
        self._base_solver.set_num_threads(num_threads)

    @property
    def temperature(self) -> float:
        return self._base_solver.get_temperature()
    
    @temperature.setter
    def temperature(self, temperature: float) -> None:
        self._base_solver.set_temperature(temperature)

    @property
    def state_update_method(self) -> StateUpdateMethod:
        return _cast_base_state_update_method(self._base_solver.get_state_update_method())
    
    @state_update_method.setter
    def state_update_method(self, state_update_method: Union[str, StateUpdateMethod]) -> None:
        self._base_solver.set_state_update_method(
            _cast_state_update_method(state_update_method)
        )

    @property
    def random_number_engine(self) -> RandomNumberEngine:
        return _cast_base_random_number_engine(self._base_solver.get_random_number_engine())
    
    @random_number_engine.setter
    def random_number_engine(self, random_number_engine: Union[str, RandomNumberEngine]) -> None:
        self._base_solver.set_random_number_engine(
            _cast_random_number_engine(random_number_engine)
        )

    @property
    def spin_selection_method(self) -> SpinSelectionMethod:
        return _cast_base_spin_selection_method(self._base_solver.get_spin_selection_method())
    
    @spin_selection_method.setter
    def spin_selection_method(self, spin_selection_method: Union[str, SpinSelectionMethod]) -> None:
        self._base_solver.set_spin_selection_method(
            _cast_spin_selection_method(spin_selection_method)
        )