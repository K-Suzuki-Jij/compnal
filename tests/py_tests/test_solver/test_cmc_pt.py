import itertools as it

import numpy as np

from compnal.lattice import (
    BoundaryCondition,
    Chain,
    Cubic,
    InfiniteRange,
    LatticeType,
    Square,
)
from compnal.model.classical import ClassicalModelType, Ising, PolyIsing
from compnal.solver import CMC
from compnal.solver.parameters import (
    CMCAlgorithm,
    RandomNumberEngine,
    SpinSelectionMethod,
    StateUpdateMethod,
    TemperatureDistribution,
)


def test_cmc_pt_ising_chain():
    boundary_condition_list = [BoundaryCondition.OBC, BoundaryCondition.PBC]
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for bc, sum, rne, ssm, td in all_list:
        chain = Chain(system_size=5, boundary_condition=bc)
        ising = Ising(
            lattice=chain,
            linear=1.0,
            quadratic=-1.0,
            spin_magnitude=0.5,
            spin_scale_factor=2,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 5)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (0,): 0,
                (1,): 1,
                (2,): 2,
                (3,): 3,
                (4,): 4,
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.ISING
            assert result.model_info.interactions == {1: 1.0, 2: -1.0}
            assert result.model_info.spin_magnitude == {
                (0,): 0.5,
                (1,): 0.5,
                (2,): 0.5,
                (3,): 0.5,
                (4,): 0.5,
            }
            assert result.model_info.spin_scale_factor == 2
            assert result.model_info.lattice.lattice_type == LatticeType.CHAIN
            assert result.model_info.lattice.system_size == 5
            assert result.model_info.lattice.shape == (5,)
            assert result.model_info.lattice.boundary_condition == bc

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_ising_square():
    boundary_condition_list = [BoundaryCondition.OBC, BoundaryCondition.PBC]
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for bc, sum, rne, ssm, td in all_list:
        square = Square(x_size=4, y_size=3, boundary_condition=bc)
        ising = Ising(
            lattice=square,
            linear=1.0,
            quadratic=-1.0,
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 12)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (i, j): j * 4 + i for i in range(4) for j in range(3)
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.ISING
            assert result.model_info.interactions == {1: 1.0, 2: -1.0}
            assert result.model_info.spin_magnitude == {
                (i, j): 1 for i in range(4) for j in range(3)
            }
            assert result.model_info.spin_scale_factor == 1
            assert result.model_info.lattice.lattice_type == LatticeType.SQUARE
            assert result.model_info.lattice.system_size == 12
            assert result.model_info.lattice.shape == (4, 3)
            assert result.model_info.lattice.boundary_condition == bc

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_ising_cubic():
    boundary_condition_list = [BoundaryCondition.OBC, BoundaryCondition.PBC]
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for bc, sum, rne, ssm, td in all_list:
        cubic = Cubic(x_size=3, y_size=4, z_size=3, boundary_condition=bc)
        ising = Ising(
            lattice=cubic,
            linear=10.0,
            quadratic=-1.0,
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 36)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (i, j, k): k * 12 + j * 3 + i
                for i in range(3)
                for j in range(4)
                for k in range(3)
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.ISING
            assert result.model_info.interactions == {1: 10.0, 2: -1.0}
            assert result.model_info.spin_magnitude == {
                (i, j, k): 1 for i in range(3) for j in range(4) for k in range(3)
            }
            assert result.model_info.spin_scale_factor == 1
            assert result.model_info.lattice.lattice_type == LatticeType.CUBIC
            assert result.model_info.lattice.system_size == 36
            assert result.model_info.lattice.shape == (3, 4, 3)
            assert result.model_info.lattice.boundary_condition == bc

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_ising_infinite_range():
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for sum, rne, ssm, td in all_list:
        infinite_range = InfiniteRange(system_size=5)
        ising = Ising(
            lattice=infinite_range,
            linear=1.0,
            quadratic=-1.0,
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 5)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (0,): 0,
                (1,): 1,
                (2,): 2,
                (3,): 3,
                (4,): 4,
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.ISING
            assert result.model_info.interactions == {1: 1.0, 2: -1.0}
            assert result.model_info.spin_magnitude == {
                (0,): 1,
                (1,): 1,
                (2,): 1,
                (3,): 1,
                (4,): 1,
            }
            assert result.model_info.spin_scale_factor == 1
            assert result.model_info.lattice.lattice_type == LatticeType.INFINITE_RANGE
            assert result.model_info.lattice.system_size == 5
            assert result.model_info.lattice.shape == None
            assert (
                result.model_info.lattice.boundary_condition == BoundaryCondition.NONE
            )

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_poly_ising_chain():
    boundary_condition_list = [BoundaryCondition.OBC, BoundaryCondition.PBC]
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for bc, sum, rne, ssm, td in all_list:
        chain = Chain(system_size=5, boundary_condition=bc)
        ising = PolyIsing(
            lattice=chain,
            interaction={1: -1, 4: +2},
            spin_magnitude=0.5,
            spin_scale_factor=2,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 5)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (0,): 0,
                (1,): 1,
                (2,): 2,
                (3,): 3,
                (4,): 4,
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.POLY_ISING
            assert result.model_info.interactions == {1: -1, 4: +2}
            assert result.model_info.spin_magnitude == {
                (0,): 0.5,
                (1,): 0.5,
                (2,): 0.5,
                (3,): 0.5,
                (4,): 0.5,
            }
            assert result.model_info.spin_scale_factor == 2
            assert result.model_info.lattice.lattice_type == LatticeType.CHAIN
            assert result.model_info.lattice.system_size == 5
            assert result.model_info.lattice.shape == (5,)
            assert result.model_info.lattice.boundary_condition == bc

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_poly_ising_square():
    boundary_condition_list = [BoundaryCondition.OBC, BoundaryCondition.PBC]
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for bc, sum, rne, ssm, td in all_list:
        square = Square(x_size=4, y_size=3, boundary_condition=bc)
        ising = PolyIsing(
            lattice=square,
            interaction={1: -1, 2: +2},
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 12)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (i, j): j * 4 + i for i in range(4) for j in range(3)
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.POLY_ISING
            assert result.model_info.interactions == {1: -1, 3: +2}
            assert result.model_info.spin_magnitude == {
                (i, j): 1 for i in range(4) for j in range(3)
            }
            assert result.model_info.spin_scale_factor == 1
            assert result.model_info.lattice.lattice_type == LatticeType.SQUARE
            assert result.model_info.lattice.system_size == 12
            assert result.model_info.lattice.shape == (4, 3)
            assert result.model_info.lattice.boundary_condition == bc

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_poly_ising_cubic():
    boundary_condition_list = [BoundaryCondition.OBC, BoundaryCondition.PBC]
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for bc, sum, rne, ssm, td in all_list:
        cubic = Cubic(x_size=3, y_size=4, z_size=3, boundary_condition=bc)
        ising = PolyIsing(
            lattice=cubic,
            interaction={1: -1, 2: +2},
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 36)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (i, j, k): k * 12 + j * 3 + i
                for i in range(3)
                for j in range(4)
                for k in range(3)
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.POLY_ISING
            assert result.model_info.interactions == {1: -1, 3: +2}
            assert result.model_info.spin_magnitude == {
                (i, j, k): 1 for i in range(3) for j in range(4) for k in range(3)
            }
            assert result.model_info.spin_scale_factor == 1
            assert result.model_info.lattice.lattice_type == LatticeType.CUBIC
            assert result.model_info.lattice.system_size == 36
            assert result.model_info.lattice.shape == (3, 4, 3)
            assert result.model_info.lattice.boundary_condition == bc

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)


def test_cmc_pt_poly_ising_infinite_range():
    state_update_method_list = [
        StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.HEAT_BATH,
    ]
    random_number_engine_list = [
        RandomNumberEngine.MT,
        RandomNumberEngine.MT_64,
        RandomNumberEngine.XORSHIFT,
    ]
    spin_selection_method_list = [
        SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.SEQUENTIAL,
    ]
    temperature_distribution_list = [
        TemperatureDistribution.ARITHMETIC,
        TemperatureDistribution.GEOMETRIC,
    ]

    all_list = it.product(
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
        temperature_distribution_list,
    )

    for sum, rne, ssm, td in all_list:
        infinite_range = InfiniteRange(system_size=5)
        ising = PolyIsing(
            lattice=infinite_range,
            interaction={1: 1, 3: -3, 4: +2},
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_parallel_tempering(
            model=ising,
            temperature_range=(0.5, 1.0),
            num_sweeps=20,
            replica_exchange_ratio=0.1,
            num_replicas=2,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            temperature_distribution=td,
            seed=0,
        )

        assert len(result_set) == 2

        for i, result in enumerate(result_set):
            assert result.samples.shape == (1, 5)
            assert result.samples.dtype == np.int8
            assert result.energies.shape == (1,)
            assert result.coordinate_to_index == {
                (0,): 0,
                (1,): 1,
                (2,): 2,
                (3,): 3,
                (4,): 4,
            }
            assert result.temperature == 0.5 + 0.5 * i

            assert result.model_info.model_type == ClassicalModelType.POLY_ISING
            assert result.model_info.interactions == {1: 1, 3: -3, 4: +2}
            assert result.model_info.spin_magnitude == {
                (0,): 1,
                (1,): 1,
                (2,): 1,
                (3,): 1,
                (4,): 1,
            }
            assert result.model_info.spin_scale_factor == 1
            assert result.model_info.lattice.lattice_type == LatticeType.INFINITE_RANGE
            assert result.model_info.lattice.system_size == 5
            assert result.model_info.lattice.shape == None
            assert (
                result.model_info.lattice.boundary_condition == BoundaryCondition.NONE
            )

            assert result.params.num_sweeps == 20
            assert result.params.num_samples == 1
            assert result.params.num_threads == 1
            assert result.params.num_replicas == 2
            assert result.params.num_replica_exchange == 2
            assert result.params.state_update_method == sum
            assert result.params.random_number_engine == rne
            assert result.params.spin_selection_method == ssm
            assert result.params.algorithm == CMCAlgorithm.PARALLEL_TEMPERING
            assert result.params.seed == 0

            assert isinstance(result.hardware_info.cpu_threads, int)
            assert isinstance(result.hardware_info.cpu_cores, int)
            assert isinstance(result.hardware_info.cpu_name, str)
            assert isinstance(result.hardware_info.memory_size, float)
            assert isinstance(result.hardware_info.os_info, str)

            assert isinstance(result.time.date, str)
            assert isinstance(result.time.total, float)
            assert isinstance(result.time.sample, float)
            assert isinstance(result.time.energy, float)
