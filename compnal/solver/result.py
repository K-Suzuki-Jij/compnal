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


from typing import Optional
from dataclasses import dataclass, asdict
from compnal.model.classical.model_info import ClassicalModelInfo
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod
)
import numpy as np


@dataclass
class CMCHardwareInfo:
    """Class for the hardware information of the classical Monte Carlo solver.

    Attributes:
        date (str, optional): Date. Defaults to None.
        total_time (float, optional): Total time [sec]. Defaults to None.
        sampling_time (float, optional): Sampling time [sec]. Defaults to None.
        cpu_threads (int, optional): Number of CPU threads. Defaults to None.
        cpu_cores (int, optional): Number of CPU cores. Defaults to None.
        cpu_name (str, optional): CPU name. Defaults to None.
        memory_size (int, optional): Memory size [GB]. Defaults to None.
        os_info (str, optional): OS information. Defaults to None.
    """
    date: Optional[str] = None
    total_time: Optional[float] = None
    sampling_time: Optional[float] = None
    cpu_threads: Optional[int] = None
    cpu_cores: Optional[int] = None
    cpu_name: Optional[str] = None
    memory_size: Optional[int] = None
    os_info: Optional[str] = None

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return asdict(self)
    
    @classmethod
    def from_serializable(cls, obj: dict):
        """Convert from a serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            CMCHardwareInfo: Hardware information.
        """
        return cls(**obj)
    
@dataclass
class CMCParams:
    """Class for the parameters of the classical Monte Carlo solver.

    Attributes:
        num_sweeps (int, Optional): Number of sweeps. Defaults to None.
        num_samples (int, Optional): Number of samples. Defaults to None.
        num_threads (int, Optional): Number of threads. Defaults to None.
        state_update_method (StateUpdateMethod, Optional): State update method. Defaults to None.
        random_number_engine (RandomNumberEngine, Optional): Random number engine. Defaults to None.
        spin_selection_method (SpinSelectionMethod, Optional): Spin selection method. Defaults to None.
        seed (int): Seed. Defaults to None.
    """
    num_sweeps: Optional[int] = None
    num_samples: Optional[int] = None
    num_threads: Optional[int] = None
    state_update_method: Optional[StateUpdateMethod] = None
    random_number_engine: Optional[RandomNumberEngine] = None
    spin_selection_method: Optional[SpinSelectionMethod] = None
    seed: Optional[int] = None

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return asdict(self)
    
    @classmethod
    def from_serializable(cls, obj: dict):
        """Convert from a serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            CMCParams: Parameters.
        """
        return cls(**obj)


@dataclass
class CMCResult:
    """Class for the results of the classical Monte Carlo solver.

    Attributes:
        samples (list[dict], Optional): Samples. Defaults to None.
        energies (list[float], Optional): Energies. Defaults to None.
        temperature (float, Optional): Temperature. Defaults to None.
        model_info (ClassicalModelInfo, Optional): Model information. Defaults to None.
        hard_info (CMCHardwareInfo, Optional): Hardware information. Defaults to None.
        params (CMCParams, Optional): Parameters. Defaults to None.
    """
    samples: Optional[list[dict]] = None
    energies: Optional[list[float]] = None
    temperature: Optional[float] = None
    model_info: Optional[ClassicalModelInfo] = None
    hard_info: Optional[CMCHardwareInfo] = None
    params: Optional[CMCParams] = None

    def calculate_mean(self, bias: float = 0.0) -> float:
        """Calculate the mean of the samples.

        Args:
            bias (float, optional): The bias in E(X - bias). Defaults to 0.0.

        Returns:
            float: The mean of the samples.
        """
        return self.calculate_moment(order=1, bias=bias)

    def calculate_moment(self, order: int, bias: float = 0.0) -> float:
        """Calculate the moment of the samples.

        Args:
            order (int): The order of the moment.
            bias (float, optional): The bias in E((X - bias)^order). Defaults to 0.0.

        Returns:
            float: The moment of the samples.
        """
        return np.mean(
            [np.mean(np.array(list(sample.values())) + bias)**order for sample in self.samples]
        )

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return {
            "samples_values": [list(sample.values()) for sample in self.samples],
            "samples_keys": [list(sample.keys()) for sample in self.samples],
            "energies": self.energies,
            "temperature": self.temperature,
            "model_info": self.model_info.to_serializable(),
            "hard_info": self.hard_info.to_serializable(),
            "params": self.params.to_serializable()
        }

    @classmethod
    def from_serializable(cls, obj: dict):
        """Convert from a serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            CMCResult: Results.
        """
        return cls(
            samples=[
                dict(zip(
                    obj["samples_keys"][i],
                    obj["samples_values"][i]
                )) for i in range(len(obj["samples_keys"]))
            ],
            energies=obj["energies"],
            temperature=obj["temperature"],
            model_info=ClassicalModelInfo.from_serializable(obj["model_info"]),
            hard_info=CMCHardwareInfo.from_serializable(obj["hard_info"]),
            params=CMCParams.from_serializable(obj["params"])
        )
    







    







    



