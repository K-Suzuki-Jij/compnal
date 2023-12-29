import numpy as np

from compnal.lattice import (
    BoundaryCondition,
    Chain,
    Cubic,
    InfiniteRange,
    LatticeInfo,
    LatticeType,
    Square,
)
from compnal.model.classical import ClassicalModelInfo, ClassicalModelType, PolyIsing
from compnal.solver import (
    CMC,
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
    result_set.export_hdf5("./test.hdf5")

    new_result_set = CMCResultSet.import_hdf5("./test.hdf5")
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


def test_cmc_result_h5py_from_ssf():
    t_list = np.linspace(0.1, 1.0, 30)
    L = 5
    p = 3
    num_sweeps = 5
    num_samples = 2
    lattice_list = [
        InfiniteRange(system_size=L),
        Square(x_size=L, y_size=L, boundary_condition="OBC"),
        Chain(system_size=L, boundary_condition="OBC"),
        Cubic(x_size=L, y_size=L, z_size=L, boundary_condition="OBC"),
    ]

    for LAT in lattice_list:
        result = CMCResultSet()
        for t in t_list:
            lattice = LAT
            ising = PolyIsing(
                lattice, {p: -1.0}, spin_magnitude=0.5, spin_scale_factor=2
            )
            result.merge(
                CMC.run_single_flip(
                    model=ising,
                    temperature=t,
                    num_sweeps=num_sweeps,
                    num_samples=num_samples,
                    num_threads=32,
                )
            )
        result.export_hdf5("./test.hdf5")

        from_result = CMCResultSet.import_hdf5("./test.hdf5")

        assert len(result) == len(from_result)
        assert (result[0].samples == from_result[0].samples).all()
        assert (result[0].energies == from_result[0].energies).all()
        assert result[0].coordinate_to_index == from_result[0].coordinate_to_index
        assert result[0].temperature == from_result[0].temperature

        assert result[0].model_info.model_type == from_result[0].model_info.model_type
        assert (
            result[0].model_info.interactions == from_result[0].model_info.interactions
        )
        assert (
            result[0].model_info.spin_magnitude
            == from_result[0].model_info.spin_magnitude
        )
        assert (
            result[0].model_info.spin_scale_factor
            == from_result[0].model_info.spin_scale_factor
        )

        assert (
            result[0].model_info.lattice.lattice_type
            == from_result[0].model_info.lattice.lattice_type
        )
        assert (
            result[0].model_info.lattice.system_size
            == from_result[0].model_info.lattice.system_size
        )
        assert (
            result[0].model_info.lattice.shape
            == from_result[0].model_info.lattice.shape
        )
        assert (
            result[0].model_info.lattice.boundary_condition
            == from_result[0].model_info.lattice.boundary_condition
        )

        assert result[0].params.num_sweeps == from_result[0].params.num_sweeps
        assert result[0].params.num_samples == from_result[0].params.num_samples
        assert result[0].params.num_threads == from_result[0].params.num_threads
        assert result[0].params.num_replicas == from_result[0].params.num_replicas
        assert (
            result[0].params.num_replica_exchange
            == from_result[0].params.num_replica_exchange
        )
        assert (
            result[0].params.state_update_method
            == from_result[0].params.state_update_method
        )
        assert (
            result[0].params.random_number_engine
            == from_result[0].params.random_number_engine
        )
        assert (
            result[0].params.spin_selection_method
            == from_result[0].params.spin_selection_method
        )
        assert result[0].params.algorithm == from_result[0].params.algorithm
        assert result[0].params.seed == from_result[0].params.seed

        assert (
            result[0].hardware_info.cpu_threads
            == from_result[0].hardware_info.cpu_threads
        )
        assert (
            result[0].hardware_info.cpu_cores == from_result[0].hardware_info.cpu_cores
        )
        assert result[0].hardware_info.cpu_name == from_result[0].hardware_info.cpu_name
        assert (
            result[0].hardware_info.memory_size
            == from_result[0].hardware_info.memory_size
        )
        assert result[0].hardware_info.os_info == from_result[0].hardware_info.os_info

        assert result[0].time.date == from_result[0].time.date
        assert result[0].time.total == from_result[0].time.total
        assert result[0].time.sample == from_result[0].time.sample
        assert result[0].time.energy == from_result[0].time.energy
