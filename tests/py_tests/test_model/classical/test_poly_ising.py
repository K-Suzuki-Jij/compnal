import pytest
import numpy as np
from compnal.model.classical import PolyIsing
from compnal.lattice import Chain, Square, Cubic, InfiniteRange


def test_poly_ising_chain():
    chain = Chain(system_size=4, boundary_condition="OBC")
    poly_ising = PolyIsing(lattice=chain, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0})
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3

    with pytest.raises(ValueError):
        poly_ising.calculate_energy([-1, +1])

    with pytest.raises(ValueError):
        poly_ising.calculate_energy([-1, +1, -1, +1, +1])

    with pytest.raises(ValueError):
        PolyIsing(lattice=chain, interaction={-1: -1.0})

    chain = Chain(system_size=3, boundary_condition="OBC")
    assert PolyIsing(chain, {}).calculate_energy([-1,+1,-1]) == 0.0
    assert PolyIsing(chain, {0: 3.0}).calculate_energy([-1,+1,-1]) == +3.0
    assert PolyIsing(chain, {1: 1.0}).calculate_energy([-1,+1,-1]) == -1.0
    assert PolyIsing(chain, {2: 2.0}).calculate_energy([-1,+1,-1]) == -4.0
    assert PolyIsing(chain, {3: 2.0}).calculate_energy([-1,+1,-1]) == +2.0
    assert PolyIsing(chain, {1: 1.0, 3: 2.0}).calculate_energy([-1,+1,-1]) == +1.0

    chain = Chain(system_size=3, boundary_condition="PBC")
    assert PolyIsing(chain, {}).calculate_energy([-1,+1,-1]) == 0.0
    assert PolyIsing(chain, {0: 3.0}).calculate_energy([-1,+1,-1]) == +3.0
    assert PolyIsing(chain, {1: 1.0}).calculate_energy([-1,+1,-1]) == -1.0
    assert PolyIsing(chain, {2: 2.0}).calculate_energy([-1,+1,-1]) == -2.0
    assert PolyIsing(chain, {3: 2.0}).calculate_energy([-1,+1,-1]) == +6.0
    assert PolyIsing(chain, {1: 1.0, 3: 2.0}).calculate_energy([-1,+1,-1]) == +5.0

def test_poly_ising_square():
    square = Square(x_size=4, y_size=3, boundary_condition="OBC")
    poly_ising = PolyIsing(lattice=square, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0})
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3

    with pytest.raises(ValueError):
        poly_ising.calculate_energy(np.reshape([-1,+1,-1,+1,+1,+1], (2, 3)))

    with pytest.raises(ValueError):
        poly_ising.calculate_energy(np.zeros((4, 4)))

    with pytest.raises(ValueError):
        PolyIsing(lattice=square, interaction={-1: -1.0})

    spins = np.reshape([-1,+1,-1,+1,+1,+1], (3, 2))
    square = Square(x_size=3, y_size=2, boundary_condition="OBC")
    assert PolyIsing(square, {}).calculate_energy(spins) == 0.0
    assert PolyIsing(square, {0: 3.0}).calculate_energy(spins) == +3.0
    assert PolyIsing(square, {1: 1.0}).calculate_energy(spins) == +2.0
    assert PolyIsing(square, {2: 2.0}).calculate_energy(spins) == -2.0
    assert PolyIsing(square, {3: 2.0}).calculate_energy(spins) == +4.0
    assert PolyIsing(square, {0: 3.0, 1: 2.0, 3: 2.0}).calculate_energy(spins) == +11.0

    square = Square(x_size=3, y_size=2, boundary_condition="PBC")
    assert PolyIsing(square, {}).calculate_energy(spins) == 0.0
    assert PolyIsing(square, {0: 3.0}).calculate_energy(spins) == +3.0
    assert PolyIsing(square, {1: 1.0}).calculate_energy(spins) == +2.0
    assert PolyIsing(square, {2: 2.0}).calculate_energy(spins) == +0.0
    assert PolyIsing(square, {3: 2.0}).calculate_energy(spins) == +16.0
    assert PolyIsing(square, {0: 3.0, 1: 2.0, 3: 2.0}).calculate_energy(spins) == +23.0

def test_poly_ising_cubic():
    cubic = Cubic(x_size=4, y_size=3, z_size=2, boundary_condition="OBC")
    poly_ising = PolyIsing(lattice=cubic, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0})
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3

    with pytest.raises(ValueError):
        poly_ising.calculate_energy(np.reshape([-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1], (2, 3, 2)))

    with pytest.raises(ValueError):
        poly_ising.calculate_energy(np.zeros((4, 4, 4)))

    with pytest.raises(ValueError):
        PolyIsing(lattice=cubic, interaction={-1: -1.0})

    spins = np.reshape([-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1], (3, 2, 2))
    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="OBC")
    assert PolyIsing(cubic, {}).calculate_energy(spins) == 0.0
    assert PolyIsing(cubic, {0: 3.0}).calculate_energy(spins) == +3.0
    assert PolyIsing(cubic, {1: 1.0}).calculate_energy(spins) == +4.0
    assert PolyIsing(cubic, {2: 2.0}).calculate_energy(spins) == +8.0
    assert PolyIsing(cubic, {3: 2.0}).calculate_energy(spins) == +8.0
    assert PolyIsing(cubic, {1: 1.0, 3: 2.0}).calculate_energy(spins) == +12.0

    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="PBC")
    assert PolyIsing(cubic, {}).calculate_energy(spins) == 0.0
    assert PolyIsing(cubic, {0: 3.0}).calculate_energy(spins) == +3.0
    assert PolyIsing(cubic, {1: 1.0}).calculate_energy(spins) == +4.0
    assert PolyIsing(cubic, {2: 2.0}).calculate_energy(spins) == +24.0
    assert PolyIsing(cubic, {3: 2.0}).calculate_energy(spins) == +40.0
    assert PolyIsing(cubic, {1: 1.0, 3: 2.0}).calculate_energy(spins) == +44.0

def test_poly_ising_infinite_range():
    infinite_range = InfiniteRange(system_size=4)
    poly_ising = PolyIsing(lattice=infinite_range, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0})
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3

    with pytest.raises(ValueError):
        poly_ising.calculate_energy([+1,+1,+1,+1,+1])

    with pytest.raises(ValueError):
        poly_ising.calculate_energy([+1,+1,+1])

    with pytest.raises(ValueError):
        PolyIsing(lattice=infinite_range, interaction={-1: -1.0})

    infinite_range = InfiniteRange(system_size=3)
    assert PolyIsing(infinite_range, {}).calculate_energy([-1,+1,-1]) == 0.0
    assert PolyIsing(infinite_range, {0: 3.0}).calculate_energy([-1,+1,-1]) == +3.0
    assert PolyIsing(infinite_range, {1: 1.0}).calculate_energy([-1,+1,-1]) == -1.0
    assert PolyIsing(infinite_range, {2: 2.0}).calculate_energy([-1,+1,-1]) == -2.0
    assert PolyIsing(infinite_range, {3: 3.0}).calculate_energy([-1,+1,-1]) == +3.0
    assert PolyIsing(infinite_range, {1: 1.0, 2: 2.0, 3: 3.0}).calculate_energy([-1,+1,-1]) == +0.0
