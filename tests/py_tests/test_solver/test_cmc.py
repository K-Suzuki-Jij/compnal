import pytest
import statistics
from compnal.model.classical import Ising
from compnal.lattice import Chain, Square, Cubic, InfiniteRange
from compnal.solver import CMC
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod
)

def test_cmc_ising_chain():
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
                    results = CMC.run_sampling(
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
                    for sample in results.samples:
                        assert statistics.mean(sample.values()) == pytest.approx(-1.0, abs=1e-8)

                    assert results.temperature == 1.0
                    assert results.params.num_sweeps == 1000
                    assert results.params.num_samples == 1
                    assert results.params.num_threads == 1
                    assert results.params.state_update_method == state_update_method
                    assert results.params.random_number_engine == random_number_engine
                    assert results.params.spin_selection_method == spin_selection_method
                    assert results.params.seed == 0


def test_cmc_ising_square():
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
                    results = CMC.run_sampling(
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
                    for sample in results.samples:
                        assert statistics.mean(sample.values()) == pytest.approx(-1.0, abs=1e-8)

                    assert results.temperature == 0.5
                    assert results.params.num_sweeps == 2000
                    assert results.params.num_samples == 1
                    assert results.params.num_threads == 1
                    assert results.params.state_update_method == state_update_method
                    assert results.params.random_number_engine == random_number_engine
                    assert results.params.spin_selection_method == spin_selection_method
                    assert results.params.seed == 0


def test_cmc_ising_cubic():
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
                        lattice=cubic, linear=1.0, quadratic=-1.0, spin_magnitude=1, spin_scale_factor=1
                    )
                    results = CMC.run_sampling(
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
                    for sample in results.samples:
                        assert statistics.mean(sample.values()) == pytest.approx(-1.0, abs=1e-8)

                    assert results.temperature == 0.8
                    assert results.params.num_sweeps == 4000
                    assert results.params.num_samples == 1
                    assert results.params.num_threads == 1
                    assert results.params.state_update_method == state_update_method
                    assert results.params.random_number_engine == random_number_engine
                    assert results.params.spin_selection_method == spin_selection_method
                    assert results.params.seed == 0


def test_cmc_ising_infinite_range():
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
                results = CMC.run_sampling(
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
                for sample in results.samples:
                    assert statistics.mean(sample.values()) == pytest.approx(-1.0, abs=1e-8)

                assert results.temperature == 1.0
                assert results.params.num_sweeps == 1000
                assert results.params.num_samples == 1
                assert results.params.num_threads == 1
                assert results.params.state_update_method == state_update_method
                assert results.params.random_number_engine == random_number_engine
                assert results.params.spin_selection_method == spin_selection_method
                assert results.params.seed == 0