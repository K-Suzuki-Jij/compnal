import pytest
import numpy as np
from compnal.model.classical import Ising
from compnal.lattice import Chain, Square, Cubic, InfiniteRange


def test_ising_chain():
    chain = Chain(system_size=3, boundary_condition="OBC")
    ising = Ising(lattice=chain, linear=1.0, quadratic=-4.0)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == -4.0
    assert ising.calculate_energy(np.array([-1, +1, -1])) == 7.0

    with pytest.raises(ValueError):
        ising.calculate_energy(np.array([-1, +1]))

    with pytest.raises(ValueError):
        ising.calculate_energy(np.array([-1, +1, -1, +1]))

    chain = Chain(system_size=3, boundary_condition="PBC")
    ising = Ising(lattice=chain, linear=1.0, quadratic=-4.0)
    assert ising.calculate_energy(np.array([-1, +1, -1])) == 3.0


def test_ising_square():
    square = Square(x_size=3, y_size=2, boundary_condition="OBC")
    ising = Ising(lattice=square, linear=1.0, quadratic=2.0)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.calculate_energy(np.reshape([-1,+1,-1,+1,+1,+1], (3, 2))) == 0.0

    with pytest.raises(ValueError):
        ising.calculate_energy(np.array([[-1, +1], [+1, -1]]))

    with pytest.raises(ValueError):
        ising.calculate_energy(np.array([[-1, +1], [+1, -1], [-1, +1], [+1, -1]]))

    square = Square(x_size=3, y_size=2, boundary_condition="PBC")
    ising = Ising(lattice=square, linear=1.0, quadratic=2.0)
    assert ising.calculate_energy(np.array([[-1, +1], [-1, +1], [+1, +1]])) == 2.0


def test_ising_cubic():
    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="OBC")
    ising = Ising(lattice=cubic, linear=1.0, quadratic=2.0)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.calculate_energy(
        np.reshape([-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1], (3, 2, 2))
    ) == 12.0

    with pytest.raises(ValueError):
        ising.calculate_energy([[[-1, +1], [-1, +1]], [[+1, +1], [+1, +1]]])

    with pytest.raises(ValueError):
        ising.calculate_energy(
            [[[-1, +1], [-1, +1]], [[+1, +1], [+1, +1]], [[-1, +1], [-1, +1]], [[+1, +1], [+1, +1]]]
        )

    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="PBC")
    ising = Ising(lattice=cubic, linear=1.0, quadratic=2.0)
    assert ising.calculate_energy(
        np.reshape([-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1], (3, 2, 2))
    ) == 28.0

def test_ising_infinite_range():
    infinite_range = InfiniteRange(system_size=3)
    ising = Ising(lattice=infinite_range, linear=1.0, quadratic=2.0)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.calculate_energy(np.array([-1,+1,-1])) == -3.0

    with pytest.raises(ValueError):
        ising.calculate_energy(np.array([-1, +1]))

    with pytest.raises(ValueError):
        ising.calculate_energy(np.array([-1, +1, -1, +1]))
