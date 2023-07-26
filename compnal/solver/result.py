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


import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import numpy as np

from compnal.base_compnal import base_utility
from compnal.model.classical.model_info import ClassicalModelInfo
from compnal.solver.parameters import (
    CMCAlgorithm,
    RandomNumberEngine,
    SpinSelectionMethod,
    StateUpdateMethod,
)


@dataclass
class CMCTime:
    """Class for the time information of the classical Monte Carlo solver.

    Attributes:
        date (str, optional): Date. Defaults to None.
        total (float, optional): Total time [sec]. Defaults to None.
        sample (float, optional): Sampling time [sec]. Defaults to None.
        energy (float, optional): Energy calculation time [sec]. Defaults to None.
    """

    date: Optional[str] = None
    total: Optional[float] = None
    sample: Optional[float] = None
    energy: Optional[float] = None

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
class CMCHardwareInfo:
    """Class for the hardware information of the classical Monte Carlo solver.

    Attributes:
        cpu_threads (int, optional): Number of CPU threads. Defaults to None.
        cpu_cores (int, optional): Number of CPU cores. Defaults to None.
        cpu_name (str, optional): CPU name. Defaults to None.
        memory_size (float, optional): Memory size [GB]. Defaults to None.
        os_info (str, optional): OS information. Defaults to None.
    """

    cpu_threads: Optional[int] = None
    cpu_cores: Optional[int] = None
    cpu_name: Optional[str] = None
    memory_size: Optional[float] = None
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
        num_replicas (int, Optional): Number of replicas. Defaults to None.
        num_replica_exchange (int, Optional): Number of replica exchange. Defaults to None.
        state_update_method (StateUpdateMethod, Optional): State update method. Defaults to None.
        random_number_engine (RandomNumberEngine, Optional): Random number engine. Defaults to None.
        spin_selection_method (SpinSelectionMethod, Optional): Spin selection method. Defaults to None.
        algorithm (CMCAlgorithm, Optional): Algorithm. Defaults to None.
        seed (int): Seed. Defaults to None.
    """

    num_sweeps: Optional[int] = None
    num_samples: Optional[int] = None
    num_threads: Optional[int] = None
    num_replicas: Optional[int] = None
    num_replica_exchange: Optional[int] = None
    state_update_method: Optional[StateUpdateMethod] = None
    random_number_engine: Optional[RandomNumberEngine] = None
    spin_selection_method: Optional[SpinSelectionMethod] = None
    algorithm: Optional[CMCAlgorithm] = None
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
        samples (list[list[float]], Optional): Samples. Defaults to None.
        energies (list[float], Optional): Energies. Defaults to None.
        coordinate_to_index (dict[Union[int, tuple], int], Optional): Coordinate to index. Defaults to None.
        temperature (float, Optional): Temperature. Defaults to None.
        model_info (ClassicalModelInfo, Optional): Model information. Defaults to None.
        params (CMCParams, Optional): Parameters. Defaults to None.
        hardware_info (CMCHardwareInfo): Hardware information. Defaults to None.
        time (CMCTime): Time information. Defaults to None.
    """

    samples: Optional[list[list[float]]] = None
    energies: Optional[list[float]] = None
    coordinate_to_index: Optional[dict[Union[int, tuple], int]] = None
    temperature: Optional[float] = None
    model_info: Optional[ClassicalModelInfo] = None
    params: Optional[CMCParams] = None
    hardware_info: Optional[CMCHardwareInfo] = None
    time: Optional[CMCTime] = None

    def calculate_mean(self, bias: float = 0.0) -> tuple[float, float]:
        """Calculate the mean of the samples.

        Args:
            bias (float, optional): The bias in E(X - bias). Defaults to 0.0.

        Returns:
            Union[tuple[float, float], float]: The mean of the samples, and if `std` is `True`, its standard deviation.
        """
        return self.calculate_moment(order=1, bias=bias)

    def calculate_moment(self, order: int, bias: float = 0.0) -> tuple[float, float]:
        """Calculate the moment of the samples.

        Args:
            order (int): The order of the moment.
            bias (float, optional): The bias in E((X - bias)^order). Defaults to 0.0.

        Returns:
            float: The moment of the samples.
        """
        return base_utility.calculate_moment(
            self.samples, order=order, bias=bias, num_threads=self.params.num_threads
        )

    def calculate_correlation(
        self, i: Union[int, tuple], j: Union[int, tuple]
    ) -> tuple[float, float]:
        """Calculate the correlation between the spin i and j.

        Args:
            i (Union[int, tuple]): The coordinate of the spin.
            j (Union[int, tuple]): The coordinate of the spin.

        Returns:
            float: The correlation between the spin i and j.
        """
        return np.mean(
            self.samples.T[self.coordinate_to_index[i]]
            * self.samples.T[self.coordinate_to_index[j]]
        )

    def calculate_energy(self, std=False) -> tuple[float, float]:
        """Calculate the energy.

        Args:
            std (bool, optional): If True, calculate the standard deviation. Defaults to False.

        Returns:
            Union[tuple[float, float], float]: The energy, and if `std` is `True`, its standard deviation.
        """
        if std:
            return np.mean(self.energies), np.std(self.energies)
        else:
            return np.mean(self.energies)

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return {
            "samples": [list(sample) for sample in self.samples],
            "energies": list(self.energies),
            "coordinate": list(self.coordinate_to_index.keys()),
            "temperature": self.temperature,
            "model_info": self.model_info.to_serializable(),
            "params": self.params.to_serializable(),
            "hardware_info": self.hardware_info.to_serializable(),
            "time": self.time.to_serializable(),
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
            samples=np.array(obj["samples"]),
            energies=np.array(obj["energies"]),
            coordinate_to_index={
                tuple(coo): i for i, coo in enumerate(obj["coordinate"])
            },
            temperature=obj["temperature"],
            model_info=ClassicalModelInfo.from_serializable(obj["model_info"]),
            params=CMCParams.from_serializable(obj["params"]),
            hardware_info=CMCHardwareInfo.from_serializable(obj["hardware_info"]),
            time=CMCTime.from_serializable(obj["time"]),
        )


@dataclass
class CMCResultSet:
    results: dict[uuid.UUID, CMCResult] = field(default_factory=dict)
    index_to_uuid: list[uuid.UUID] = field(default_factory=list)

    def append(self, result: CMCResult) -> None:
        """Append the result.

        Args:
            result (CMCResult): Result.
        """
        new_uuid = uuid.uuid4()
        self.index_to_uuid.append(new_uuid)
        self.results[new_uuid] = result

    def merge(self, other) -> None:
        """Merge the results.

        Args:
            other (CMCResultSet): Results.
        """
        self.results.update(other.results)
        self.index_to_uuid.extend(other.index_to_uuid)

    def to_serializable(self) -> dict:
        """Convert to a serializable object.

        Returns:
            dict: Serializable object.
        """
        return {
            "results": {
                str(key): value.to_serializable() for key, value in self.results.items()
            },
            "index_to_uuid": [str(value) for value in self.index_to_uuid],
        }

    @classmethod
    def from_serializable(cls, obj: dict):
        """Convert from a serializable object.

        Args:
            obj (dict): Serializable object.

        Returns:
            CMCResultSet: Results.
        """
        return cls(
            results={
                uuid.UUID(key): CMCResult.from_serializable(value)
                for key, value in obj["results"].items()
            },
            index_to_uuid=[uuid.UUID(value) for value in obj["index_to_uuid"]],
        )

    def __len__(self) -> int:
        """Return the number of results.

        Returns:
            int: The number of results.
        """
        return len(self.results)

    def __getitem__(self, index: int) -> CMCResult:
        """Return the result.

        Args:
            index (int): Index.

        Returns:
            CMCResult: Result.
        """
        return self.results[self.index_to_uuid[index]]

    def __iter__(self):
        return iter(self.results.values())

    def __next__(self):
        return next(self.results.values())
