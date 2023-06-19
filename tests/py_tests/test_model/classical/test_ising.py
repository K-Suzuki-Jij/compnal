import pytest
import numpy as np
from compnal.model.classical import Ising
from compnal.lattice import Chain, Square, Cubic, InfiniteRange


def test_ising_chain():
    chain = Chain(system_size=3, boundary_condition="OBC")
    ising = Ising(lattice=chain, linear=1.0, quadratic=-4.0, spin_magnitude=1)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == -4.0
    assert ising.get_spin_magnitude() == {0: 1, 1: 1, 2: 1}
    assert ising.get_spin_scale_factor() == 1
    ising.set_spin_magnitude(2, 0)
    assert ising.get_spin_magnitude() == {0: 2, 1: 1, 2: 1}

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, 0)
    
    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, 10)

    with pytest.raises(ValueError):
        Ising(lattice=chain, linear=1.0, quadratic=-4.0, spin_magnitude=1.5, spin_scale_factor=0)

    chain = Chain(system_size=3, boundary_condition="PBC")
    ising = Ising(lattice=chain, linear=1.0, quadratic=-4.0)


def test_ising_square():
    square = Square(x_size=3, y_size=2, boundary_condition="OBC")
    ising = Ising(lattice=square, linear=1.0, quadratic=2.0)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {(0,0): 0.5, (1,0): 0.5, (2,0): 0.5, (0,1): 0.5, (1,1): 0.5, (2,1): 0.5}
    assert ising.get_spin_scale_factor() == 1
    ising.set_spin_magnitude(2, (0,1))
    assert ising.get_spin_magnitude() == {(0,0): 0.5, (1,0): 0.5, (2,0): 0.5, (0,1): 2, (1,1): 0.5, (2,1): 0.5}

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, (0,0))
    
    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, (10,0))

    with pytest.raises(ValueError):
        Ising(lattice=square, linear=1.0, quadratic=-4.0, spin_magnitude=1.5, spin_scale_factor=0)

    square = Square(x_size=3, y_size=2, boundary_condition="PBC")
    ising = Ising(lattice=square, linear=1.0, quadratic=2.0)


def test_ising_cubic():
    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="OBC")
    ising = Ising(lattice=cubic, linear=1.0, quadratic=2.0, spin_magnitude=3, spin_scale_factor=2)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {
        (0,0,0): 3, (1,0,0): 3, (2,0,0): 3, (0,1,0): 3, (1,1,0): 3, (2,1,0): 3,
        (0,0,1): 3, (1,0,1): 3, (2,0,1): 3, (0,1,1): 3, (1,1,1): 3, (2,1,1): 3,
    }
    assert ising.get_spin_scale_factor() == 2
    ising.set_spin_magnitude(2, (0,0,1))
    assert ising.get_spin_magnitude() == {
        (0,0,0): 3, (1,0,0): 3, (2,0,0): 3, (0,1,0): 3, (1,1,0): 3, (2,1,0): 3,
        (0,0,1): 2, (1,0,1): 3, (2,0,1): 3, (0,1,1): 3, (1,1,1): 3, (2,1,1): 3,
    }

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, (0,0,0))
    
    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, (10,0,0))

    with pytest.raises(ValueError):
        Ising(lattice=cubic, linear=1.0, quadratic=-4.0, spin_magnitude=1.5, spin_scale_factor=0)

    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="PBC")
    ising = Ising(lattice=cubic, linear=1.0, quadratic=2.0)


def test_ising_infinite_range():
    infinite_range = InfiniteRange(system_size=3)
    ising = Ising(lattice=infinite_range, linear=1.0, quadratic=2.0, spin_magnitude=1, spin_scale_factor=3)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {0: 1, 1: 1, 2: 1}
    assert ising.get_spin_scale_factor() == 3
    ising.set_spin_magnitude(2, 0)
    assert ising.get_spin_magnitude() == {0: 2, 1: 1, 2: 1}

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, 0)
    
    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, 10)

    with pytest.raises(ValueError):
        Ising(lattice=infinite_range, linear=1.0, quadratic=-4.0, spin_magnitude=1.5, spin_scale_factor=0)
