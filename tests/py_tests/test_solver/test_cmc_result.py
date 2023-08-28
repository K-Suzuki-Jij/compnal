import numpy as np

from compnal.lattice import BoundaryCondition, LatticeInfo, LatticeType
from compnal.model.classical import ClassicalModelInfo, ClassicalModelType
from compnal.solver import (
    CMCAlgorithm,
    CMCHardwareInfo,
    CMCParams,
    CMCResult,
    CMCResultSet,
    CMCTime,
    RandomNumberEngine,
    SpinSelectionMethod,
    StateUpdateMethod,
)


def test_cmc_time():
    time = CMCTime.from_serializable(
        CMCTime(
            date="2021-08-01",
            total=200,
            sample=100,
            energy=10,
        ).to_serializable()
    )

    assert time.date == "2021-08-01"
    assert time.total == 200
    assert time.sample == 100
    assert time.energy == 10


def test_cmc_hardware_info():
    info = CMCHardwareInfo.from_serializable(
        CMCHardwareInfo(
            cpu_threads=12,
            cpu_cores=6,
            cpu_name="Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
            memory_size=16,
            os_info="Linux-5.4.0-80-generic-x86_64-with-glibc2.29",
        ).to_serializable()
    )

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
            algorithm=CMCAlgorithm.PARALLEL_TEMPERING,
            seed=0,
        ).to_serializable()
    )

    assert params.num_sweeps == 100
    assert params.num_samples == 1000
    assert params.num_threads == 12
    assert params.num_replicas == None
    assert params.num_replica_exchange == None
    assert params.state_update_method == StateUpdateMethod.METROPOLIS
    assert params.random_number_engine == RandomNumberEngine.MT
    assert params.spin_selection_method == SpinSelectionMethod.RANDOM
    assert params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
    assert params.seed == 0


def test_cmc_result_serialize():
    result = CMCResult.from_serializable(
        CMCResult(
            samples=np.full((4, 2), -1),
            energies=np.array([1, 2, 3, 4]),
            coordinate_to_index={(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},
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
                ),
            ),
            params=CMCParams(
                num_sweeps=100,
                num_samples=1000,
                num_threads=12,
                state_update_method=StateUpdateMethod.METROPOLIS,
                random_number_engine=RandomNumberEngine.MT,
                spin_selection_method=SpinSelectionMethod.RANDOM,
                algorithm=CMCAlgorithm.PARALLEL_TEMPERING,
                seed=0,
            ),
            hardware_info=CMCHardwareInfo(
                cpu_threads=12,
                cpu_cores=6,
                cpu_name="Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
                memory_size=16,
                os_info="Linux-5.4.0-80-generic-x86_64-with-glibc2.29",
            ),
            time=CMCTime(
                date="2021-08-01",
                total=200,
                sample=100,
                energy=10,
            ),
        ).to_serializable()
    )

    assert (result.samples == np.full((4, 2), -1)).all()
    assert (result.energies == np.array([1, 2, 3, 4])).all()
    assert result.coordinate_to_index == {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    assert result.temperature == 1.0
    assert result.model_info.model_type == ClassicalModelType.ISING
    assert result.model_info.interactions == {0: 1.0, 2: 1.0}
    assert result.model_info.spin_magnitude == {
        (0, 0): 0.5,
        (0, 1): 0.5,
        (1, 0): 0.5,
        (1, 1): 0.5,
    }
    assert result.model_info.spin_scale_factor == 2
    assert result.model_info.lattice.lattice_type == LatticeType.SQUARE
    assert result.model_info.lattice.system_size == 4
    assert result.model_info.lattice.shape == (2, 2)
    assert result.model_info.lattice.boundary_condition == BoundaryCondition.OBC
    assert result.params.num_sweeps == 100
    assert result.params.num_samples == 1000
    assert result.params.num_threads == 12
    assert result.params.num_replicas == None
    assert result.params.num_replica_exchange == None
    assert result.params.state_update_method == StateUpdateMethod.METROPOLIS
    assert result.params.random_number_engine == RandomNumberEngine.MT
    assert result.params.spin_selection_method == SpinSelectionMethod.RANDOM
    assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
    assert result.params.seed == 0
    assert result.hardware_info.cpu_threads == 12
    assert result.hardware_info.cpu_cores == 6
    assert result.hardware_info.cpu_name == "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"
    assert result.hardware_info.memory_size == 16
    assert (
        result.hardware_info.os_info == "Linux-5.4.0-80-generic-x86_64-with-glibc2.29"
    )
    assert result.time.date == "2021-08-01"
    assert result.time.total == 200
    assert result.time.sample == 100
    assert result.time.energy == 10

def test_cmc_result_h5py():
    result_set = CMCResultSet()
    result_set.append(
        CMCResult(
            samples=np.full((4, 2), -1),
            energies=np.array([1, 2, 3, 4]),
            coordinate_to_index={(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},
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
                ),
            ),
            params=CMCParams(
                num_sweeps=100,
                num_samples=1000,
                num_threads=12,
                state_update_method=StateUpdateMethod.METROPOLIS,
                random_number_engine=RandomNumberEngine.MT,
                spin_selection_method=SpinSelectionMethod.RANDOM,
                algorithm=CMCAlgorithm.PARALLEL_TEMPERING,
                seed=0,
            ),
            hardware_info=CMCHardwareInfo(
                cpu_threads=12,
                cpu_cores=6,
                cpu_name="Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
                memory_size=16,
                os_info="Linux-5.4.0-80-generic-x86_64-with-glibc2.29",
            ),
            time=CMCTime(
                date="2021-08-01",
                total=200,
                sample=100,
                energy=10,
            ),
        )
    )
    result_set.export_as_hdf5("./test.hdf5")

    new_result_set = CMCResultSet.import_from_hdf5("./test.hdf5")
    result = new_result_set[0]

    assert (result.samples == np.full((4, 2), -1)).all()
    assert (result.energies == np.array([1, 2, 3, 4])).all()
    assert result.coordinate_to_index == {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    assert result.temperature == 1.0
    assert result.model_info.model_type == ClassicalModelType.ISING
    assert result.model_info.interactions == {0: 1.0, 2: 1.0}
    assert result.model_info.spin_magnitude == {
        (0, 0): 0.5,
        (0, 1): 0.5,
        (1, 0): 0.5,
        (1, 1): 0.5,
    }
    assert result.model_info.spin_scale_factor == 2
    assert result.model_info.lattice.lattice_type == LatticeType.SQUARE
    assert result.model_info.lattice.system_size == 4
    assert result.model_info.lattice.shape == (2, 2)
    assert result.model_info.lattice.boundary_condition == BoundaryCondition.OBC
    assert result.params.num_sweeps == 100
    assert result.params.num_samples == 1000
    assert result.params.num_threads == 12
    assert result.params.num_replicas == None
    assert result.params.num_replica_exchange == None
    assert result.params.state_update_method == StateUpdateMethod.METROPOLIS
    assert result.params.random_number_engine == RandomNumberEngine.MT
    assert result.params.spin_selection_method == SpinSelectionMethod.RANDOM
    assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
    assert result.params.seed == 0
    assert result.hardware_info.cpu_threads == 12
    assert result.hardware_info.cpu_cores == 6
    assert result.hardware_info.cpu_name == "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"
    assert result.hardware_info.memory_size == 16
    assert (
        result.hardware_info.os_info == "Linux-5.4.0-80-generic-x86_64-with-glibc2.29"
    )
    assert result.time.date == "2021-08-01"
    assert result.time.total == 200
    assert result.time.sample == 100
    assert result.time.energy == 10