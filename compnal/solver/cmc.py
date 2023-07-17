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
from compnal.solver.result import CMCTime, CMCHardwareInfo, CMCParams, CMCResult, CMCResultSet
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod,
    TemperatureDistribution,
    CMCAlgorithm,
    _to_base_state_update_method,
    _to_base_random_number_engine,
    _to_base_spin_selection_method,
    _to_state_update_method,
    _to_random_number_engine,
    _to_spin_selection_method,
    _to_temperature_distribution
)
import numpy as np
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
    ) -> CMCResultSet:
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
            CMCResultSet: Results.
        """
        start_total_time = time.time()

        # Set seed
        if seed is None:
            seed = random.randint(0, 9223372036854775807)

        # Set solver
        _base_solver = base_solver.make_classical_monte_carlo(model._base_model)

        # Sampling
        start_sampling_time = time.time()
        samples = _base_solver.run_single_flip(
            model=model._base_model,
            num_sweeps=num_sweeps,
            num_samples=num_samples,
            num_threads=num_threads,
            temperature=temperature,
            seed=seed,
            updater=_to_base_state_update_method(state_update_method),
            random_number_engine=_to_base_random_number_engine(random_number_engine),
            spin_selector=_to_base_spin_selection_method(spin_selection_method)
        )
        end_sampling_time = time.time()

        # Calculate energies
        start_energy_time = time.time()
        energies = _base_solver.calculate_energies(
            model=model._base_model, 
            samples=samples, 
            num_threads=num_threads
        )
        end_energy_time = time.time()

        # Make coordinate list
        coordinate_list = {coo: i for i, coo in enumerate(model._lattice.generate_coordinate_list())}

        # Store parameter information
        cmc_params = CMCParams(
            num_sweeps=num_sweeps,
            num_samples=num_samples,
            num_threads=num_threads,
            state_update_method=_to_state_update_method(state_update_method),
            random_number_engine=_to_random_number_engine(random_number_engine),
            spin_selection_method=_to_spin_selection_method(spin_selection_method),
            algorithm=CMCAlgorithm.SINGLE_FLIP,
            seed=seed
        )

        # Store hardware information
        cmc_hard_info = CMCHardwareInfo(
            cpu_threads=psutil.cpu_count(),
            cpu_cores=psutil.cpu_count(logical=False),
            cpu_name=platform.processor(),
            memory_size=psutil.virtual_memory().total/(1024**3),
            os_info=platform.platform()
        )

        # Store time information
        cmc_time = CMCTime(
            date=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            total=time.time() - start_total_time,
            sample=end_sampling_time - start_sampling_time,
            energy=end_energy_time - start_energy_time,
        )

        result_set = CMCResultSet()
        result_set.append(
            CMCResult(
                samples=samples,
                energies=energies,
                coordinate_to_index=coordinate_list,
                temperature=temperature,
                model_info=model.export_info(),
                hardware_info=cmc_hard_info,
                params=cmc_params,
                time=cmc_time
            )
        )

        return result_set
    
    @staticmethod
    def run_parallel_tempering(
        model: Union[Ising, PolyIsing],
        time_range: tuple[float, float],
        num_sweeps: int = 1000,
        num_swaps: int = 1000,
        num_replicas: int = 100,
        num_samples: int = 1,
        num_threads: int = 1,
        state_update_method: Union[str, StateUpdateMethod] = "METROPOLIS",
        random_number_engine: Union[str, RandomNumberEngine] = "MT",
        spin_selection_method: Union[str, SpinSelectionMethod] = "RANDOM",
        temperature_distribution: Union[str, TemperatureDistribution] = "ARITHMETIC",
        seed: Optional[int] = None
    ) -> CMCResultSet:        

        start_total_time = time.time()
        
        # Set seed
        if seed is None:
            seed = random.randint(0, 9223372036854775807)

        # Set solver
        _base_solver = base_solver.make_classical_monte_carlo(model._base_model)

        # Generate temperature list
        if _to_temperature_distribution(temperature_distribution) == TemperatureDistribution.ARITHMETIC:
            temperature_list = np.linspace(time_range[0], time_range[1], num_replicas)
        elif _to_temperature_distribution(temperature_distribution) == TemperatureDistribution.GEOMETRIC:
            temperature_list = np.geomspace(time_range[0], time_range[1], num_replicas)
        else:
            raise ValueError("Invalid temperature distribution.")

        start_sampling_time = time.time()
        samples = _base_solver.run_parallel_tempering(
            model=model._base_model,
            num_sweeps=num_sweeps,
            num_swaps=num_swaps,
            num_samples=num_samples,
            num_threads=num_threads,
            temperature_list=temperature_list,
            seed=seed,
            updater=_to_base_state_update_method(state_update_method),
            random_number_engine=_to_base_random_number_engine(random_number_engine),
            spin_selector=_to_base_spin_selection_method(spin_selection_method)
        )
        end_sampling_time = time.time()

        samples = samples.reshape(num_replicas, num_samples, -1)

        # Calculate energies
        energies_list = []
        energy_time_list = []
        for replica_count in range(num_replicas):
            start_energy_time = time.time()
            energies_list.append(_base_solver.calculate_energies(
                    model=model._base_model, 
                    samples=samples[replica_count], 
                    num_threads=num_threads
                )
            )
            energy_time_list.append(time.time() - start_energy_time)

        # Make coordinate list
        coordinate_list = {coo: i for i, coo in enumerate(model._lattice.generate_coordinate_list())}

        # Store parameter information
        cmc_params = CMCParams(
            num_sweeps=num_sweeps,
            num_samples=num_samples,
            num_threads=num_threads,
            state_update_method=_to_state_update_method(state_update_method),
            random_number_engine=_to_random_number_engine(random_number_engine),
            spin_selection_method=_to_spin_selection_method(spin_selection_method),
            algorithm=CMCAlgorithm.PARALLEL_TEMPERING,
            seed=seed
        )

        # Store hardware information
        cmc_hard_info = CMCHardwareInfo(
            cpu_threads=psutil.cpu_count(),
            cpu_cores=psutil.cpu_count(logical=False),
            cpu_name=platform.processor(),
            memory_size=psutil.virtual_memory().total/(1024**3),
            os_info=platform.platform()
        )

        result_set = CMCResultSet()
        for replica_count in range(num_replicas):
            # Store time information
            cmc_time = CMCTime(
                date=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                total=time.time() - start_total_time,
                sample=end_sampling_time - start_sampling_time,
                energy=energy_time_list[replica_count]
            )

            result_set.append(
                CMCResult(
                    samples=samples[replica_count],
                    energies=energies_list[replica_count],
                    coordinate_to_index=coordinate_list,
                    temperature=temperature_list[replica_count],
                    model_info=model.export_info(),
                    hardware_info=cmc_hard_info,
                    params=cmc_params,
                    time=cmc_time
                )
            )

        return result_set
            
