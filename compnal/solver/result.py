#  Copyright 2024 Kohei Suzuki
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

import h5py
import numpy as np

from compnal.base_compnal import base_utility
from compnal.lattice.boundary_condition import BoundaryCondition
from compnal.lattice.lattice_info import LatticeInfo, LatticeType
from compnal.model.classical.model_info import ClassicalModelInfo, ClassicalModelType
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

    def calculate_mean(
        self, bias: float = 0.0, with_std: bool = False, num_threads: int = 1
    ) -> Union[float, tuple[float, float]]:
        """Calculate the mean of the samples.

        Args:
            bias (float, optional): The bias in E(X - bias). Defaults to 0.0.
            with_std (bool, optional): If `True`, return the standard deviation. Defaults to False.
            num_threads (int, optional): Number of threads. Defaults to 1.

        Returns:
            Union[float, tuple[float, float]]: The mean of the samples, and if `std` is `True`, its standard deviation.
        """
        return self.calculate_moment(order=1, bias=bias, with_std=with_std, num_threads=num_threads)

    def calculate_moment(
        self, order: int, bias: float = 0.0, with_std: bool = False, num_threads: int = 1
    ) -> Union[float, tuple[float, float]]:
        """Calculate the moment of the samples.

        Args:
            order (int): The order of the moment.
            bias (float, optional): The bias in E((X - bias)^order). Defaults to 0.0.
            with_std (bool, optional): If `True`, return the standard deviation. Defaults to False.
            num_threads (int, optional): Number of threads. Defaults to 1.

        Returns:
            Union[float, tuple[float, float]]: The moment of the samples, and if `std` is `True`, its standard deviation.
        """
        if with_std:
            return base_utility.calculate_moment_with_std(
                self.samples,
                order=order,
                bias=bias,
                num_threads=num_threads,
            )
        else:
            return base_utility.calculate_moment(
                self.samples,
                order=order,
                bias=bias,
                num_threads=num_threads,
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

    def calculate_energy(self) -> float:
        """Calculate the energy.

        Returns:
            float: The energy.
        """
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

    def sort(self, key: callable):
        """
        Sort the results in-place using a custom key function.

        Args:
            key (callable): A function that takes a CMCResult object and returns a value to sort by.
        """
        sorted_items = sorted(self.results.items(), key=lambda item: key(item[1]))
        self.results = dict(sorted_items)
        self.index_to_uuid = [uuid for uuid, _ in sorted_items]

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

    def export_hdf5(self, path: str) -> None:
        """Export and store the results as HDF5.

        Args:
            path (str): Path.
        """
        with h5py.File(path, "w") as f:
            group = f.create_group("CMCResultSet/")
            group.create_dataset(
                "index_to_uuid", data=[str(value) for value in self.index_to_uuid]
            )

            group = f.create_group("CMCResultSet/results/")
            for key, result in self.results.items():
                # Main results
                group.create_dataset(str(key) + "/samples", data=result.samples)
                group.create_dataset(str(key) + "/energies", data=result.energies)
                group.create_dataset(
                    str(key) + "/coordinate_to_index/keys",
                    data=list(result.coordinate_to_index.keys()),
                )
                group.create_dataset(
                    str(key) + "/coordinate_to_index/values",
                    data=list(result.coordinate_to_index.values()),
                )
                group.create_dataset(str(key) + "/temperature", data=result.temperature)

                # ClassicalModelInfo
                group.create_dataset(
                    str(key) + "/model_info/model_type",
                    data=result.model_info.model_type,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(
                    str(key) + "/model_info/interactions/keys",
                    data=list(result.model_info.interactions.keys()),
                )
                group.create_dataset(
                    str(key) + "/model_info/interactions/values",
                    data=list(result.model_info.interactions.values()),
                )
                group.create_dataset(
                    str(key) + "/model_info/spin_magnitude/keys",
                    data=list(result.model_info.spin_magnitude.keys()),
                )
                group.create_dataset(
                    str(key) + "/model_info/spin_magnitude/values",
                    data=list(result.model_info.spin_magnitude.values()),
                )
                group.create_dataset(
                    str(key) + "/model_info/spin_scale_factor",
                    data=result.model_info.spin_scale_factor,
                )

                # ClassicalModelInfo/LatticeInfo
                group.create_dataset(
                    str(key) + "/model_info/lattice/lattice_type",
                    data=result.model_info.lattice.lattice_type,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(
                    str(key) + "/model_info/lattice/system_size",
                    data=result.model_info.lattice.system_size,
                )

                if result.model_info.lattice.shape is not None:
                    group.create_dataset(
                        str(key) + "/model_info/lattice/shape",
                        data=result.model_info.lattice.shape,
                    )

                group.create_dataset(
                    str(key) + "/model_info/lattice/boundary_condition",
                    data=result.model_info.lattice.boundary_condition,
                    dtype=h5py.string_dtype(),
                )

                # CMCParams
                group.create_dataset(
                    str(key) + "/params/num_sweeps", data=result.params.num_sweeps
                )
                group.create_dataset(
                    str(key) + "/params/num_samples", data=result.params.num_samples
                )
                group.create_dataset(
                    str(key) + "/params/num_threads", data=result.params.num_threads
                )

                if result.params.num_replicas is not None:
                    group.create_dataset(
                        str(key) + "/params/num_replicas",
                        data=result.params.num_replicas,
                    )
                if result.params.num_replica_exchange is not None:
                    group.create_dataset(
                        str(key) + "/params/num_replica_exchange",
                        data=result.params.num_replica_exchange,
                    )
                group.create_dataset(
                    str(key) + "/params/state_update_method",
                    data=result.params.state_update_method,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(
                    str(key) + "/params/random_number_engine",
                    data=result.params.random_number_engine,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(
                    str(key) + "/params/spin_selection_method",
                    data=result.params.spin_selection_method,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(
                    str(key) + "/params/algorithm",
                    data=result.params.algorithm,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(str(key) + "/params/seed", data=result.params.seed)

                # CMCHardwareInfo
                group.create_dataset(
                    str(key) + "/hardware_info/cpu_threads",
                    data=result.hardware_info.cpu_threads,
                )
                group.create_dataset(
                    str(key) + "/hardware_info/cpu_cores",
                    data=result.hardware_info.cpu_cores,
                )
                group.create_dataset(
                    str(key) + "/hardware_info/cpu_name",
                    data=result.hardware_info.cpu_name,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(
                    str(key) + "/hardware_info/memory_size",
                    data=result.hardware_info.memory_size,
                )
                group.create_dataset(
                    str(key) + "/hardware_info/os_info",
                    data=result.hardware_info.os_info,
                    dtype=h5py.string_dtype(),
                )

                # CMCTime
                group.create_dataset(
                    str(key) + "/time/date",
                    data=result.time.date,
                    dtype=h5py.string_dtype(),
                )
                group.create_dataset(str(key) + "/time/total", data=result.time.total)
                group.create_dataset(str(key) + "/time/sample", data=result.time.sample)
                group.create_dataset(str(key) + "/time/energy", data=result.time.energy)

    @classmethod
    def import_hdf5(cls, path: str) -> None:
        """Import and load the results from HDF5.

        Args:
            path (str): Path.
        """
        file_contents = h5py.File(path, "r")
        index_to_uuid = [
            uuid.UUID(value.decode("utf-8"))
            for value in file_contents["CMCResultSet/index_to_uuid"]
        ]
        new_result_set = cls(
            results={
                universally_unique_id: CMCResult(
                    samples=file_contents[
                        "CMCResultSet/results/"
                        + str(universally_unique_id)
                        + "/samples"
                    ][...],
                    energies=file_contents[
                        "CMCResultSet/results/"
                        + str(universally_unique_id)
                        + "/energies"
                    ][...],
                    coordinate_to_index={
                        tuple(key): value
                        for key, value in zip(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/coordinate_to_index/keys"
                            ],
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/coordinate_to_index/values"
                            ],
                        )
                    },
                    temperature=float(
                        file_contents[
                            "CMCResultSet/results/"
                            + str(universally_unique_id)
                            + "/temperature"
                        ][...]
                    ),
                    model_info=ClassicalModelInfo(
                        model_type=ClassicalModelType(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/model_info/model_type"
                            ].asstr()[...]
                        ),
                        interactions={
                            key: value
                            for key, value in zip(
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/interactions/keys"
                                ],
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/interactions/values"
                                ],
                            )
                        },
                        spin_magnitude={
                            tuple(key): value
                            for key, value in zip(
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/spin_magnitude/keys"
                                ],
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/spin_magnitude/values"
                                ],
                            )
                        },
                        spin_scale_factor=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/model_info/spin_scale_factor"
                            ][...]
                        ),
                        lattice=LatticeInfo(
                            lattice_type=LatticeType(
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/lattice/lattice_type"
                                ].asstr()[...]
                            ),
                            system_size=int(
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/lattice/system_size"
                                ][...]
                            ),
                            shape=tuple(
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/lattice/shape"
                                ]
                            )
                            if "shape"
                            in file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/model_info/lattice"
                            ]
                            else None,
                            boundary_condition=BoundaryCondition(
                                file_contents[
                                    "CMCResultSet/results/"
                                    + str(universally_unique_id)
                                    + "/model_info/lattice/boundary_condition"
                                ].asstr()[...]
                            ),
                        ),
                    ),
                    params=CMCParams(
                        num_sweeps=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/num_sweeps"
                            ][...]
                        ),
                        num_samples=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/num_samples"
                            ][...]
                        ),
                        num_threads=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/num_threads"
                            ][...]
                        ),
                        num_replicas=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/num_replicas"
                            ][...]
                        )
                        if "num_replicas"
                        in file_contents[
                            "CMCResultSet/results/"
                            + str(universally_unique_id)
                            + "/params"
                        ]
                        else None,
                        num_replica_exchange=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/num_replica_exchange"
                            ][...]
                        )
                        if "num_replica_exchange"
                        in file_contents[
                            "CMCResultSet/results/"
                            + str(universally_unique_id)
                            + "/params"
                        ]
                        else None,
                        state_update_method=StateUpdateMethod(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/state_update_method"
                            ].asstr()[...]
                        ),
                        random_number_engine=RandomNumberEngine(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/random_number_engine"
                            ].asstr()[...]
                        ),
                        spin_selection_method=SpinSelectionMethod(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/spin_selection_method"
                            ].asstr()[...]
                        ),
                        algorithm=CMCAlgorithm(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/algorithm"
                            ].asstr()[...]
                        ),
                        seed=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/params/seed"
                            ][...]
                        ),
                    ),
                    hardware_info=CMCHardwareInfo(
                        cpu_threads=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/hardware_info/cpu_threads"
                            ][...]
                        ),
                        cpu_cores=int(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/hardware_info/cpu_cores"
                            ][...]
                        ),
                        cpu_name=str(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/hardware_info/cpu_name"
                            ].asstr()[...]
                        ),
                        memory_size=float(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/hardware_info/memory_size"
                            ][...]
                        ),
                        os_info=str(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/hardware_info/os_info"
                            ].asstr()[...]
                        ),
                    ),
                    time=CMCTime(
                        date=str(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/time/date"
                            ].asstr()[...]
                        ),
                        total=float(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/time/total"
                            ][...]
                        ),
                        sample=float(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/time/sample"
                            ][...]
                        ),
                        energy=float(
                            file_contents[
                                "CMCResultSet/results/"
                                + str(universally_unique_id)
                                + "/time/energy"
                            ][...]
                        ),
                    ),
                )
                for universally_unique_id in index_to_uuid
            },
            index_to_uuid=index_to_uuid,
        )
        file_contents.close()

        return new_result_set

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
