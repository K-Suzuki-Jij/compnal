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
)


def test_cmc_ssf_ising_chain():
    boundary_condition_list = ["OBC", "PBC"]
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

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for bc, sum, rne, ssm in all_list:
        chain = Chain(system_size=5, boundary_condition=bc)
        ising = Ising(
            lattice=chain,
            linear=1.0,
            quadratic=-1.0,
            spin_magnitude=0.5,
            spin_scale_factor=2,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=1.0,
            num_sweeps=1000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.array([[1, 1, 1, 1, 1]]),
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 5)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (0,): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
        }
        assert result_set[0].temperature == 1.0

        assert result_set[0].model_info.model_type == ClassicalModelType.ISING
        assert result_set[0].model_info.interactions == {1: 1.0, 2: -1.0}
        assert result_set[0].model_info.spin_magnitude == {
            (0,): 0.5,
            (1,): 0.5,
            (2,): 0.5,
            (3,): 0.5,
            (4,): 0.5,
        }
        assert result_set[0].model_info.spin_scale_factor == 2
        assert result_set[0].model_info.lattice.lattice_type == LatticeType.CHAIN
        assert result_set[0].model_info.lattice.system_size == 5
        assert result_set[0].model_info.lattice.shape == (5,)
        assert result_set[0].model_info.lattice.boundary_condition == bc

        assert result_set[0].params.num_sweeps == 1000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_ising_square():
    boundary_condition_list = ["OBC", "PBC"]
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

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for bc, sum, rne, ssm in all_list:
        square = Square(x_size=4, y_size=3, boundary_condition=bc)
        ising = Ising(
            lattice=square,
            linear=1.0,
            quadratic=-1.0,
            spin_magnitude=1,
            spin_scale_factor=0.7,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=0.5,
            num_sweeps=2000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.full((1, 12), 0.7)
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 12)
        assert result_set[0].samples.dtype == np.float64
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (i, j): j * 4 + i for i in range(4) for j in range(3)
        }
        assert result_set[0].temperature == 0.5

        assert result_set[0].model_info.model_type == ClassicalModelType.ISING
        assert result_set[0].model_info.interactions == {1: 1.0, 2: -1.0}
        assert result_set[0].model_info.spin_magnitude == {
            (i, j): 1 for i in range(4) for j in range(3)
        }
        assert result_set[0].model_info.spin_scale_factor == 0.7
        assert result_set[0].model_info.lattice.lattice_type == LatticeType.SQUARE
        assert result_set[0].model_info.lattice.system_size == 12
        assert result_set[0].model_info.lattice.shape == (4, 3)
        assert result_set[0].model_info.lattice.boundary_condition == bc

        assert result_set[0].params.num_sweeps == 2000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_ising_cubic():
    boundary_condition_list = ["OBC", "PBC"]
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

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for bc, sum, rne, ssm in all_list:
        cubic = Cubic(x_size=3, y_size=4, z_size=3, boundary_condition=bc)
        ising = Ising(
            lattice=cubic,
            linear=10.0,
            quadratic=-1.0,
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=0.8,
            num_sweeps=4000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.full((1, 36), 1)
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 36)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (i, j, k): k * 12 + j * 3 + i
            for i in range(3)
            for j in range(4)
            for k in range(3)
        }
        assert result_set[0].temperature == 0.8

        assert result_set[0].model_info.model_type == ClassicalModelType.ISING
        assert result_set[0].model_info.interactions == {1: 10.0, 2: -1.0}
        assert result_set[0].model_info.spin_magnitude == {
            (i, j, k): 1 for i in range(3) for j in range(4) for k in range(3)
        }
        assert result_set[0].model_info.spin_scale_factor == 1
        assert result_set[0].model_info.lattice.lattice_type == LatticeType.CUBIC
        assert result_set[0].model_info.lattice.system_size == 36
        assert result_set[0].model_info.lattice.shape == (3, 4, 3)
        assert result_set[0].model_info.lattice.boundary_condition == bc

        assert result_set[0].params.num_sweeps == 4000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_ising_infinite_range():
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

    all_list = it.product(
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for sum, rne, ssm in all_list:
        infinite_range = InfiniteRange(system_size=5)
        ising = Ising(
            lattice=infinite_range,
            linear=1.0,
            quadratic=-1.0,
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=1.0,
            num_sweeps=1000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.full((1, 5), 1)
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 5)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (0,): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
        }
        assert result_set[0].temperature == 1.0

        assert result_set[0].model_info.model_type == ClassicalModelType.ISING
        assert result_set[0].model_info.interactions == {1: 1.0, 2: -1.0}
        assert result_set[0].model_info.spin_magnitude == {
            (0,): 1,
            (1,): 1,
            (2,): 1,
            (3,): 1,
            (4,): 1,
        }
        assert result_set[0].model_info.spin_scale_factor == 1
        assert (
            result_set[0].model_info.lattice.lattice_type == LatticeType.INFINITE_RANGE
        )
        assert result_set[0].model_info.lattice.system_size == 5
        assert result_set[0].model_info.lattice.shape == None
        assert (
            result_set[0].model_info.lattice.boundary_condition
            == BoundaryCondition.NONE
        )

        assert result_set[0].params.num_sweeps == 1000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_poly_ising_chain():
    boundary_condition_list = ["OBC", "PBC"]
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

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for bc, sum, rne, ssm in all_list:
        chain = Chain(system_size=5, boundary_condition=bc)
        ising = PolyIsing(
            lattice=chain,
            interaction={1: -1, 3: +2},
            spin_magnitude=0.5,
            spin_scale_factor=2,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=1.0,
            num_sweeps=1000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.array([[1, 1, 1, 1, 1]]),
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 5)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (0,): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
        }
        assert result_set[0].temperature == 1.0

        assert result_set[0].model_info.model_type == ClassicalModelType.POLY_ISING
        assert result_set[0].model_info.interactions == {1: -1, 3: +2}
        assert result_set[0].model_info.spin_magnitude == {
            (0,): 0.5,
            (1,): 0.5,
            (2,): 0.5,
            (3,): 0.5,
            (4,): 0.5,
        }
        assert result_set[0].model_info.spin_scale_factor == 2
        assert result_set[0].model_info.lattice.lattice_type == LatticeType.CHAIN
        assert result_set[0].model_info.lattice.system_size == 5
        assert result_set[0].model_info.lattice.shape == (5,)
        assert result_set[0].model_info.lattice.boundary_condition == bc

        assert result_set[0].params.num_sweeps == 1000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_poly_ising_square():
    boundary_condition_list = ["OBC", "PBC"]
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

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for bc, sum, rne, ssm in all_list:
        square = Square(x_size=4, y_size=3, boundary_condition=bc)
        ising = PolyIsing(
            lattice=square,
            interaction={0: +3, 2: -12},
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=0.5,
            num_sweeps=2000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.full((1, 12), 1)
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 12)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (i, j): j * 4 + i for i in range(4) for j in range(3)
        }
        assert result_set[0].temperature == 0.5

        assert result_set[0].model_info.model_type == ClassicalModelType.POLY_ISING
        assert result_set[0].model_info.interactions == {0: 3.0, 2: -12.0}
        assert result_set[0].model_info.spin_magnitude == {
            (i, j): 1 for i in range(4) for j in range(3)
        }
        assert result_set[0].model_info.spin_scale_factor == 1
        assert result_set[0].model_info.lattice.lattice_type == LatticeType.SQUARE
        assert result_set[0].model_info.lattice.system_size == 12
        assert result_set[0].model_info.lattice.shape == (4, 3)
        assert result_set[0].model_info.lattice.boundary_condition == bc

        assert result_set[0].params.num_sweeps == 2000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_poly_ising_cubic():
    boundary_condition_list = ["OBC", "PBC"]
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

    all_list = it.product(
        boundary_condition_list,
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for bc, sum, rne, ssm in all_list:
        cubic = Cubic(x_size=9, y_size=9, z_size=9, boundary_condition=bc)
        ising = PolyIsing(
            lattice=cubic,
            interaction={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=0.8,
            num_sweeps=400,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.full((1, 729), 1)
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 9 * 9 * 9)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (i, j, k): k * 81 + j * 9 + i
            for i in range(9)
            for j in range(9)
            for k in range(9)
        }
        assert result_set[0].temperature == 0.8

        assert result_set[0].model_info.model_type == ClassicalModelType.POLY_ISING
        assert result_set[0].model_info.interactions == {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
        }
        assert result_set[0].model_info.spin_magnitude == {
            (i, j, k): 1 for i in range(9) for j in range(9) for k in range(9)
        }
        assert result_set[0].model_info.spin_scale_factor == 1
        assert result_set[0].model_info.lattice.lattice_type == LatticeType.CUBIC
        assert result_set[0].model_info.lattice.system_size == 9 * 9 * 9
        assert result_set[0].model_info.lattice.shape == (9, 9, 9)
        assert result_set[0].model_info.lattice.boundary_condition == bc

        assert result_set[0].params.num_sweeps == 400
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)


def test_cmc_ssf_poly_ising_infinite_range():
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

    all_list = it.product(
        state_update_method_list,
        random_number_engine_list,
        spin_selection_method_list,
    )

    for sum, rne, ssm in all_list:
        infinite_range = InfiniteRange(system_size=5)
        ising = PolyIsing(
            lattice=infinite_range,
            interaction={1: 1, 2: 2, 3: 3},
            spin_magnitude=1,
            spin_scale_factor=1,
        )
        result_set = CMC.run_single_flip(
            model=ising,
            temperature=1.0,
            num_sweeps=1000,
            num_samples=1,
            num_threads=1,
            state_update_method=sum,
            random_number_engine=rne,
            spin_selection_method=ssm,
            seed=0,
            initial_state_list=np.full((1, 5), 1)
        )

        assert len(result_set) == 1
        assert result_set[0].samples.shape == (1, 5)
        assert result_set[0].samples.dtype == np.int8
        assert result_set[0].energies.shape == (1,)
        assert result_set[0].coordinate_to_index == {
            (0,): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
        }
        assert result_set[0].temperature == 1.0

        assert result_set[0].model_info.model_type == ClassicalModelType.POLY_ISING
        assert result_set[0].model_info.interactions == {1: 1, 2: 2, 3: 3}
        assert result_set[0].model_info.spin_magnitude == {
            (0,): 1,
            (1,): 1,
            (2,): 1,
            (3,): 1,
            (4,): 1,
        }
        assert result_set[0].model_info.spin_scale_factor == 1
        assert (
            result_set[0].model_info.lattice.lattice_type == LatticeType.INFINITE_RANGE
        )
        assert result_set[0].model_info.lattice.system_size == 5
        assert result_set[0].model_info.lattice.shape == None
        assert (
            result_set[0].model_info.lattice.boundary_condition
            == BoundaryCondition.NONE
        )

        assert result_set[0].params.num_sweeps == 1000
        assert result_set[0].params.num_samples == 1
        assert result_set[0].params.num_threads == 1
        assert result_set[0].params.state_update_method == sum
        assert result_set[0].params.random_number_engine == rne
        assert result_set[0].params.spin_selection_method == ssm
        assert result_set[0].params.algorithm == CMCAlgorithm.SINGLE_FLIP
        assert result_set[0].params.seed == 0

        assert isinstance(result_set[0].hardware_info.cpu_threads, int)
        assert isinstance(result_set[0].hardware_info.cpu_cores, int)
        assert isinstance(result_set[0].hardware_info.cpu_name, str)
        assert isinstance(result_set[0].hardware_info.memory_size, float)
        assert isinstance(result_set[0].hardware_info.os_info, str)

        assert isinstance(result_set[0].time.date, str)
        assert isinstance(result_set[0].time.total, float)
        assert isinstance(result_set[0].time.sample, float)
        assert isinstance(result_set[0].time.energy, float)
