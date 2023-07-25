from compnal.lattice import LatticeType, BoundaryCondition
from compnal.model.classical import Ising, ClassicalModelType
from compnal.lattice import Chain, Square, Cubic, InfiniteRange
from compnal.solver import CMC
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod,
    CMCAlgorithm
)

def test_cmc_ssf_ising_chain():
    boundary_condition_list = ["OBC", "PBC"]
    state_update_method_list = [StateUpdateMethod.METROPOLIS, StateUpdateMethod.HEAT_BATH]
    random_number_engine_list = [RandomNumberEngine.MT, RandomNumberEngine.MT_64, RandomNumberEngine.XORSHIFT]
    spin_selection_method_list = [SpinSelectionMethod.RANDOM, SpinSelectionMethod.SEQUENTIAL]
    
    for boundary_condition in boundary_condition_list:
        for state_update_method in state_update_method_list:
            for random_number_engine in random_number_engine_list:
                for spin_selection_method in spin_selection_method_list:
                    chain = Chain(
                        system_size=5, boundary_condition=boundary_condition
                    )
                    ising = Ising(
                        lattice=chain, linear=1.0, quadratic=-1.0, spin_magnitude=0.5, spin_scale_factor=2
                    )
                    result_set = CMC.run_single_flip(
                        model=ising,
                        temperature=1.0, 
                        num_sweeps=1000, 
                        num_samples=1, 
                        num_threads=1,
                        state_update_method=state_update_method,
                        random_number_engine=random_number_engine,
                        spin_selection_method=spin_selection_method,
                        seed=0
                    )

                    assert len(result_set) == 1
                    assert result_set[0].samples.shape == (1, 5)
                    assert result_set[0].energies.shape == (1,)
                    assert result_set[0].coordinate_to_index == {(0,): 0, (1,): 1, (2,): 2, (3,): 3, (4,): 4}
                    assert result_set[0].temperature == 1.0

                    assert result_set[0].model_info.model_type == ClassicalModelType.ISING
                    assert result_set[0].model_info.interactions == {1: 1.0, 2: -1.0}
                    assert result_set[0].model_info.spin_magnitude == {(0,): 0.5, (1,): 0.5, (2,): 0.5, (3,): 0.5, (4,): 0.5}
                    assert result_set[0].model_info.spin_scale_factor == 2
                    assert result_set[0].model_info.lattice.lattice_type == LatticeType.CHAIN
                    assert result_set[0].model_info.lattice.system_size == 5
                    assert result_set[0].model_info.lattice.shape == (5,)
                    assert result_set[0].model_info.lattice.boundary_condition == boundary_condition
                    
                    assert result_set[0].params.num_sweeps == 1000
                    assert result_set[0].params.num_samples == 1
                    assert result_set[0].params.num_threads == 1
                    assert result_set[0].params.state_update_method == state_update_method
                    assert result_set[0].params.random_number_engine == random_number_engine
                    assert result_set[0].params.spin_selection_method == spin_selection_method
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
    state_update_method_list = [StateUpdateMethod.METROPOLIS, StateUpdateMethod.HEAT_BATH]
    random_number_engine_list = [RandomNumberEngine.MT, RandomNumberEngine.MT_64, RandomNumberEngine.XORSHIFT]
    spin_selection_method_list = [SpinSelectionMethod.RANDOM, SpinSelectionMethod.SEQUENTIAL]
    
    for boundary_condition in boundary_condition_list:
        for state_update_method in state_update_method_list:
            for random_number_engine in random_number_engine_list:
                for spin_selection_method in spin_selection_method_list:
                    square = Square(
                        x_size=4, y_size=3, boundary_condition=boundary_condition
                    )
                    ising = Ising(
                        lattice=square, linear=1.0, quadratic=-1.0, spin_magnitude=1, spin_scale_factor=1
                    )
                    result_set = CMC.run_single_flip(
                        model=ising,
                        temperature=0.5, 
                        num_sweeps=2000, 
                        num_samples=1, 
                        num_threads=1,
                        state_update_method=state_update_method,
                        random_number_engine=random_number_engine,
                        spin_selection_method=spin_selection_method,
                        seed=0
                    )
                    
                    assert len(result_set) == 1
                    assert result_set[0].samples.shape == (1, 12)
                    assert result_set[0].energies.shape == (1,)
                    assert result_set[0].coordinate_to_index == {(i, j): j*4 + i for i in range(4) for j in range(3)}
                    assert result_set[0].temperature == 0.5

                    assert result_set[0].model_info.model_type == ClassicalModelType.ISING
                    assert result_set[0].model_info.interactions == {1: 1.0, 2: -1.0}
                    assert result_set[0].model_info.spin_magnitude == {(i, j): 1 for i in range(4) for j in range(3)}
                    assert result_set[0].model_info.spin_scale_factor == 1
                    assert result_set[0].model_info.lattice.lattice_type == LatticeType.SQUARE
                    assert result_set[0].model_info.lattice.system_size == 12
                    assert result_set[0].model_info.lattice.shape == (4, 3)
                    assert result_set[0].model_info.lattice.boundary_condition == boundary_condition
                    
                    assert result_set[0].params.num_sweeps == 2000
                    assert result_set[0].params.num_samples == 1
                    assert result_set[0].params.num_threads == 1
                    assert result_set[0].params.state_update_method == state_update_method
                    assert result_set[0].params.random_number_engine == random_number_engine
                    assert result_set[0].params.spin_selection_method == spin_selection_method
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
    state_update_method_list = [StateUpdateMethod.METROPOLIS, StateUpdateMethod.HEAT_BATH]
    random_number_engine_list = [RandomNumberEngine.MT, RandomNumberEngine.MT_64, RandomNumberEngine.XORSHIFT]
    spin_selection_method_list = [SpinSelectionMethod.RANDOM, SpinSelectionMethod.SEQUENTIAL]
    
    for boundary_condition in boundary_condition_list:
        for state_update_method in state_update_method_list:
            for random_number_engine in random_number_engine_list:
                for spin_selection_method in spin_selection_method_list:
                    cubic = Cubic(
                        x_size=3, y_size=4, z_size=3, boundary_condition=boundary_condition
                    )
                    ising = Ising(
                        lattice=cubic, linear=10.0, quadratic=-1.0, spin_magnitude=1, spin_scale_factor=1
                    )
                    result_set = CMC.run_single_flip(
                        model=ising,
                        temperature=0.8, 
                        num_sweeps=4000, 
                        num_samples=1, 
                        num_threads=1,
                        state_update_method=state_update_method,
                        random_number_engine=random_number_engine,
                        spin_selection_method=spin_selection_method,
                        seed=0
                    )
                    
                    assert len(result_set) == 1
                    assert result_set[0].samples.shape == (1, 36)
                    assert result_set[0].energies.shape == (1,)
                    assert result_set[0].coordinate_to_index == {(i, j, k): k*12 + j*3 + i for i in range(3) for j in range(4) for k in range(3)}
                    assert result_set[0].temperature == 0.8

                    assert result_set[0].model_info.model_type == ClassicalModelType.ISING
                    assert result_set[0].model_info.interactions == {1: 10.0, 2: -1.0}
                    assert result_set[0].model_info.spin_magnitude == {(i, j, k): 1 for i in range(3) for j in range(4) for k in range(3)}
                    assert result_set[0].model_info.spin_scale_factor == 1
                    assert result_set[0].model_info.lattice.lattice_type == LatticeType.CUBIC
                    assert result_set[0].model_info.lattice.system_size == 36
                    assert result_set[0].model_info.lattice.shape == (3, 4, 3)
                    assert result_set[0].model_info.lattice.boundary_condition == boundary_condition

                    assert result_set[0].params.num_sweeps == 4000
                    assert result_set[0].params.num_samples == 1
                    assert result_set[0].params.num_threads == 1
                    assert result_set[0].params.state_update_method == state_update_method
                    assert result_set[0].params.random_number_engine == random_number_engine
                    assert result_set[0].params.spin_selection_method == spin_selection_method
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
    state_update_method_list = [StateUpdateMethod.METROPOLIS, StateUpdateMethod.HEAT_BATH]
    random_number_engine_list = [RandomNumberEngine.MT, RandomNumberEngine.MT_64, RandomNumberEngine.XORSHIFT]
    spin_selection_method_list = [SpinSelectionMethod.RANDOM, SpinSelectionMethod.SEQUENTIAL]
    
    for state_update_method in state_update_method_list:
        for random_number_engine in random_number_engine_list:
            for spin_selection_method in spin_selection_method_list:
                infinite_range = InfiniteRange(
                    system_size=5
                )
                ising = Ising(
                    lattice=infinite_range, linear=1.0, quadratic=-1.0, spin_magnitude=1, spin_scale_factor=1
                )
                result_set = CMC.run_single_flip(
                    model=ising,
                    temperature=1.0, 
                    num_sweeps=1000, 
                    num_samples=1, 
                    num_threads=1,
                    state_update_method=state_update_method,
                    random_number_engine=random_number_engine,
                    spin_selection_method=spin_selection_method,
                    seed=0
                )
                
                assert len(result_set) == 1
                assert result_set[0].samples.shape == (1, 5)
                assert result_set[0].energies.shape == (1,)
                assert result_set[0].coordinate_to_index == {(0,): 0, (1,): 1, (2,): 2, (3,): 3, (4,): 4}
                assert result_set[0].temperature == 1.0

                assert result_set[0].model_info.model_type == ClassicalModelType.ISING
                assert result_set[0].model_info.interactions == {1: 1.0, 2: -1.0}
                assert result_set[0].model_info.spin_magnitude == {(0,): 1, (1,): 1, (2,): 1, (3,): 1, (4,): 1}
                assert result_set[0].model_info.spin_scale_factor == 1
                assert result_set[0].model_info.lattice.lattice_type == LatticeType.INFINITE_RANGE
                assert result_set[0].model_info.lattice.system_size == 5
                assert result_set[0].model_info.lattice.shape == None
                assert result_set[0].model_info.lattice.boundary_condition == BoundaryCondition.NONE

                assert result_set[0].params.num_sweeps == 1000
                assert result_set[0].params.num_samples == 1
                assert result_set[0].params.num_threads == 1
                assert result_set[0].params.state_update_method == state_update_method
                assert result_set[0].params.random_number_engine == random_number_engine
                assert result_set[0].params.spin_selection_method == spin_selection_method
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