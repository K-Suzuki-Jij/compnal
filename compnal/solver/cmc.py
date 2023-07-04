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
import random


class CMC:
    """Class for the classical Monte Carlo solver.
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def run_single_flip(
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
        if seed is None:
            seed = random.randint(0, 9223372036854775807)

        start_total_time = time.time()
        _base_solver = base_solver.make_classical_monte_carlo(model._base_model)

        start_sampling_time = time.time()
        samples = _base_solver.run_single_flip(
            model=model._base_model,
            num_sweeps=num_sweeps,
            num_samples=num_samples,
            num_threads=num_threads,
            temperature=temperature,
            seed=seed,
            updater=_cast_state_update_method(state_update_method),
            random_number_engine=_cast_random_number_engine(random_number_engine),
            spin_selector=_cast_spin_selection_method(spin_selection_method)
        )
        end_sampling_time = time.time()

        energies = _base_solver.calculate_energies(model._base_model, samples, num_threads)

        coordinate_list = {coo: i for i, coo in enumerate(model._lattice.generate_coordinate_list())}

        cmc_params = CMCParams(
            num_sweeps=num_sweeps,
            num_samples=num_samples,
            num_threads=num_threads,
            state_update_method=state_update_method,
            random_number_engine=random_number_engine,
            spin_selection_method=spin_selection_method,
            seed=seed
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
            coordinate_to_index=coordinate_list,
            temperature=temperature,
            model_info=model.export_info(),
            hard_info=cmc_info,
            params=cmc_params
        )