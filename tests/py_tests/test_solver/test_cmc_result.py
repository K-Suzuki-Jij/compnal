from compnal.solver import (
    CMCResult, 
    CMCHardwareInfo, 
    CMCParams, 
    StateUpdateMethod, 
    RandomNumberEngine, 
    SpinSelectionMethod
)
from compnal.model.classical import ClassicalModelInfo, ClassicalModelType
from compnal.lattice import LatticeInfo, LatticeType, BoundaryCondition
import numpy as np

def test_cmc_hardware_info():
    info = CMCHardwareInfo.from_serializable(
        CMCHardwareInfo(
            date="20230621014801",
            total_time=1.0,
            sampling_time=0.5,
            cpu_threads=12,
            cpu_cores=6,
            cpu_name="Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
            memory_size=16,
            os_info="Linux-5.4.0-80-generic-x86_64-with-glibc2.29"
        ).to_serializable()
    )

    assert info.date == "20230621014801"
    assert info.total_time == 1.0
    assert info.sampling_time == 0.5
    assert info.cpu_threads == 12
    assert info.cpu_cores == 6
    assert info.cpu_name == "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"
    assert info.memory_size == 16
    assert info.os_info == "Linux-5.4.0-80-generic-x86_64-with-glibc2.29"
    

def test_cmc_params():
    params = CMCParams.from_serializable(
        CMCParams(
            num_sweeps=100,
            num_samples=1000,
            num_threads=12,
            state_update_method=StateUpdateMethod.METROPOLIS,
            random_number_engine=RandomNumberEngine.MT,
            spin_selection_method=SpinSelectionMethod.RANDOM,
            seed=0
        ).to_serializable()
    )

    assert params.num_sweeps == 100
    assert params.num_samples == 1000
    assert params.num_threads == 12
    assert params.state_update_method == StateUpdateMethod.METROPOLIS
    assert params.random_number_engine == RandomNumberEngine.MT
    assert params.spin_selection_method == SpinSelectionMethod.RANDOM
    assert params.seed == 0


def test_cmc_result():
    result = CMCResult.from_serializable(
        CMCResult(
            samples=np.full((2, 2), -1),
            energies=np.array([1,2,3,4]),
            coordinate=[(0, 0), (0, 1), (1, 0), (1, 1)],
            temperature=1.0,
            model_info=ClassicalModelInfo(
                model_type=ClassicalModelType.ISING,
                interactions={0: 1.0, 2: 1.0},
                spin_magnitude={(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5},
                spin_scale_factor=2,
                lattice=LatticeInfo(
                    lattice_type=LatticeType.SQUARE,
                    system_size=4,
                    shape=(2, 2),
                    boundary_condition=BoundaryCondition.OBC,
                )
            ),
            params=CMCParams(
                num_sweeps=100,
                num_samples=1000,
                num_threads=12,
                state_update_method=StateUpdateMethod.METROPOLIS,
                random_number_engine=RandomNumberEngine.MT,
                spin_selection_method=SpinSelectionMethod.RANDOM,
                seed=0
            ),
            hard_info=CMCHardwareInfo(
                date="20230621014801",
                total_time=1.0,
                sampling_time=0.5,
                cpu_threads=12,
                cpu_cores=6,
                cpu_name="Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
                memory_size=16,
                os_info="Linux-5.4.0-80-generic-x86_64-with-glibc2.29"
            )
        ).to_serializable()
    )

    assert (result.samples == np.full((2, 2), -1)).all()
    assert (result.energies == np.array([1,2,3,4])).all()
    assert result.coordinate == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert result.temperature == 1.0
    