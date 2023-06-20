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
from compnal.base_compnal import base_solver
from compnal.model.classical import Ising, PolyIsing
from compnal.solver.result import CMCHardwareInfo, CMCParams, CMCResult
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
import time
import datetime
import psutil
import platform


class CMC:
    """Class for the classical Monte Carlo solver.
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def run_sampling(
        model: Union[Ising, PolyIsing],
        temperature: float,     
        num_sweeps: int = 1000,
        num_samples: int = 1,
        num_threads: int = 1,
        state_update_method: Union[str, StateUpdateMethod] = "METROPOLIS",
        random_number_engine: Union[str, RandomNumberEngine] = "MT",
        spin_selection_method: Union[str, SpinSelectionMethod] = "RANDOM",
        seed: Optional[int] = None
    ) -> CMCResult:
        """Run sampling.

        Args:
            model (Union[Ising, PolyIsing]): Model.
            temperature (float): Temperature.
            num_sweeps (int, optional): Number of sweeps. Defaults to 1000.
            num_samples (int, optional): Number of samples. Defaults to 1.
            num_threads (int, optional): Number of threads. Defaults to 1.
            state_update_method (Union[str, StateUpdateMethod], optional): State update method. Defaults to "METROPOLIS".
            random_number_engine (Union[str, RandomNumberEngine], optional): Random number engine. Defaults to "MT".
            spin_selection_method (Union[str, SpinSelectionMethod], optional): Spin selection method. Defaults to "RANDOM".
            seed (Optional[int], optional): Seed. Defaults to None.
            
        Returns:
            CMCResult: Results.
        """
        start_total_time = time.time()
        _base_solver = base_solver.make_classical_monte_carlo(model._base_model)

        _base_solver.set_temperature(temperature)
        _base_solver.set_num_sweeps(num_sweeps)
        _base_solver.set_num_samples(num_samples)
        _base_solver.set_num_threads(num_threads)
        _base_solver.set_state_update_method(
            _cast_state_update_method(state_update_method)
        )
        _base_solver.set_random_number_engine(
            _cast_random_number_engine(random_number_engine)
        )
        _base_solver.set_spin_selection_method(
            _cast_spin_selection_method(spin_selection_method)
        )

        start_sampling_time = time.time()
        if seed is not None:
            _base_solver.run_sampling(seed)
        else:
            _base_solver.run_sampling()
        end_sampling_time = time.time()

        coordinate_list = model._lattice.generate_coordinate_list()
        samples = [dict(zip(coordinate_list, sample)) for sample in _base_solver.get_samples()]
        energies = _base_solver.calculate_energies()
        
        cmc_params = CMCParams(
            num_sweeps=num_sweeps,
            num_samples=num_samples,
            num_threads=num_threads,
            state_update_method=_cast_base_state_update_method(_base_solver.get_state_update_method()),
            random_number_engine=_cast_base_random_number_engine(_base_solver.get_random_number_engine()),
            spin_selection_method=_cast_base_spin_selection_method(_base_solver.get_spin_selection_method()),
            seed=_base_solver.get_seed()
        )

        cmc_info = CMCHardwareInfo(
            date=datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
            total_time=time.time() - start_total_time,
            sampling_time=end_sampling_time - start_sampling_time,
            cpu_threads=psutil.cpu_count(),
            cpu_cores=psutil.cpu_count(logical=False),
            cpu_name=platform.processor(),
            memory_size=psutil.virtual_memory().total/(1024**3),
            os_info=platform.platform()
        )

        return CMCResult(
            samples=samples,
            energies=energies,
            temperature=temperature,
            model_info=model.export_info(),
            hard_info=cmc_info,
            params=cmc_params
        )