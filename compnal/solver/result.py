from typing import Optional, Union
from dataclasses import dataclass
from compnal.model.classical import Ising, PolyIsing
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod
)


@dataclass
class CMCInfo:
    """Class for the information of the classical Monte Carlo solver.

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


@dataclass
class CMCResult:
    """Class for the results of the classical Monte Carlo solver.

    Attributes:
        samples (list[dict], Optional): Samples. Defaults to None.
        energies (list[float], Optional): Energies. Defaults to None.
        temperature (float, Optional): Temperature. Defaults to None.
        model (Union[Ising, PolyIsing], Optional): Model. Defaults to None.
        info (CMCInfo, Optional): Information. Defaults to None.
        params (CMCParams, Optional): Parameters. Defaults to None.
    """
    samples: Optional[list[dict]] = None
    energies: Optional[list[float]] = None
    temperature: Optional[float] = None
    model: Union[Ising, PolyIsing] = None
    info: Optional[CMCInfo] = None
    params: Optional[CMCParams] = None

    def calculate_mean(bias: float = 0.0) -> float:
        pass

    def calculate_moment(order: int, bias: float = 0.0) -> float:
        pass

    def to_serializable() -> dict:
        pass

    @classmethod
    def from_serializable():
        pass
    







    







    



