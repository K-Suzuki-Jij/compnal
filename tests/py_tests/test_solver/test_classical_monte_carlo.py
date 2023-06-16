import pytest
import numpy as np
from compnal.model.classical import Ising
from compnal.lattice import Chain, Square, Cubic, InfiniteRange
from compnal.solver import ClassicalMonteCarlo
from compnal.solver.parameters import (
    StateUpdateMethod,
    RandomNumberEngine,
    SpinSelectionMethod
)

def test_classical_monte_carlo_ising_chain():
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
                        lattice=chain, linear=1.0, quadratic=-4.0, spin_magnitude=0.5, spin_scale_factor=2
                    )
                    cmc = ClassicalMonteCarlo(model=ising)
                    samples = cmc.run_sampling(
                        temperature=1.0, 
                        num_sweeps=1000, 
                        num_samples=1, 
                        num_threads=1,
                        state_update_method=state_update_method,
                        random_number_engine=random_number_engine,
                        spin_selection_method=spin_selection_method,
                        seed=0
                    )
                    for sample in samples:
                        assert abs(np.mean(sample)) == pytest.approx(1.0, abs=1e-8)

                    assert cmc.temperature == 1.0
                    assert cmc.num_sweeps == 1000
                    assert cmc.num_samples == 1
                    assert cmc.num_threads == 1
                    assert cmc.state_update_method == state_update_method
                    assert cmc.random_number_engine == random_number_engine
                    assert cmc.spin_selection_method == spin_selection_method
                    assert cmc.get_seed() == 0


def test_classical_monte_carlo_ising_square():
    boundary_condition_list = ["OBC", "PBC"]
    state_update_method_list = [StateUpdateMethod.METROPOLIS, StateUpdateMethod.HEAT_BATH]
    random_number_engine_list = [RandomNumberEngine.MT, RandomNumberEngine.MT_64, RandomNumberEngine.XORSHIFT]
    spin_selection_method_list = [SpinSelectionMethod.RANDOM, SpinSelectionMethod.SEQUENTIAL]
    
    for boundary_condition in boundary_condition_list:
        for state_update_method in state_update_method_list:
            for random_number_engine in random_number_engine_list:
                for spin_selection_method in spin_selection_method_list:
                    square = Square(
                        x_size=3, y_size=4, boundary_condition=boundary_condition
                    )
                    ising = Ising(
                        lattice=square, linear=1.0, quadratic=-4.0, spin_magnitude=1, spin_scale_factor=1
                    )
                    cmc = ClassicalMonteCarlo(model=ising)
                    samples = cmc.run_sampling(
                        temperature=1.0, 
                        num_sweeps=2000, 
                        num_samples=1, 
                        num_threads=1,
                        state_update_method=state_update_method,
                        random_number_engine=random_number_engine,
                        spin_selection_method=spin_selection_method,
                        seed=0
                    )
                    for sample in samples:
                        assert abs(np.mean(sample)) == pytest.approx(1.0, abs=1e-8)

                    assert cmc.temperature == 1.0
                    assert cmc.num_sweeps == 2000
                    assert cmc.num_samples == 1
                    assert cmc.num_threads == 1
                    assert cmc.state_update_method == state_update_method
                    assert cmc.random_number_engine == random_number_engine
                    assert cmc.spin_selection_method == spin_selection_method
                    assert cmc.get_seed() == 0


def test_classical_monte_carlo_ising_cubic():
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
                        lattice=cubic, linear=1.0, quadratic=-4.0, spin_magnitude=1, spin_scale_factor=1
                    )
                    cmc = ClassicalMonteCarlo(model=ising)
                    samples = cmc.run_sampling(
                        temperature=2.0, 
                        num_sweeps=5000, 
                        num_samples=1, 
                        num_threads=1,
                        state_update_method=state_update_method,
                        random_number_engine=random_number_engine,
                        spin_selection_method=spin_selection_method,
                        seed=0
                    )
                    for sample in samples:
                        assert abs(np.mean(sample)) == pytest.approx(1.0, abs=1e-8)

                    assert cmc.temperature == 2.0
                    assert cmc.num_sweeps == 5000
                    assert cmc.num_samples == 1
                    assert cmc.num_threads == 1
                    assert cmc.state_update_method == state_update_method
                    assert cmc.random_number_engine == random_number_engine
                    assert cmc.spin_selection_method == spin_selection_method
                    assert cmc.get_seed() == 0


def test_classical_monte_carlo_ising_infinite_range():
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
                    lattice=infinite_range, linear=1.0, quadratic=-4.0, spin_magnitude=1, spin_scale_factor=1
                )
                cmc = ClassicalMonteCarlo(model=ising)
                samples = cmc.run_sampling(
                    temperature=1.0, 
                    num_sweeps=1000, 
                    num_samples=1, 
                    num_threads=1,
                    state_update_method=state_update_method,
                    random_number_engine=random_number_engine,
                    spin_selection_method=spin_selection_method,
                    seed=0
                )
                for sample in samples:
                    assert abs(np.mean(sample)) == pytest.approx(1.0, abs=1e-8)

                assert cmc.temperature == 1.0
                assert cmc.num_sweeps == 1000
                assert cmc.num_samples == 1
                assert cmc.num_threads == 1
                assert cmc.state_update_method == state_update_method
                assert cmc.random_number_engine == random_number_engine
                assert cmc.spin_selection_method == spin_selection_method
                assert cmc.get_seed() == 0


