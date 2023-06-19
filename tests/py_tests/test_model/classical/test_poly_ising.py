import pytest
import numpy as np
from compnal.model.classical import PolyIsing
from compnal.lattice import Chain, Square, Cubic, InfiniteRange


def test_poly_ising_chain():
    chain = Chain(system_size=4, boundary_condition="OBC")
    poly_ising = PolyIsing(lattice=chain, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=1)
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3
    assert poly_ising.get_spin_magnitude() == {0: 1, 1: 1, 2: 1, 3: 1}
    assert poly_ising.get_spin_scale_factor() == 1
    poly_ising.set_spin_magnitude(2, 0)
    assert poly_ising.get_spin_magnitude() == {0: 2, 1: 1, 2: 1, 3: 1}

    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.4999, 0)
    
    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.5, 10)

    with pytest.raises(ValueError):
        PolyIsing(lattice=chain, interaction={-1: -1.0})

    with pytest.raises(ValueError):
        PolyIsing(lattice=chain, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=1.5, spin_scale_factor=0)


def test_poly_ising_square():
    square = Square(x_size=4, y_size=3, boundary_condition="OBC")
    poly_ising = PolyIsing(lattice=square, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0})
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3
    assert poly_ising.get_spin_magnitude() == {
        (0,0): 0.5, (1,0): 0.5, (2,0): 0.5, (3,0): 0.5,
        (0,1): 0.5, (1,1): 0.5, (2,1): 0.5, (3,1): 0.5,
        (0,2): 0.5, (1,2): 0.5, (2,2): 0.5, (3,2): 0.5
    }
    poly_ising.set_spin_magnitude(2, (0,1))
    assert poly_ising.get_spin_magnitude() == {
        (0,0): 0.5, (1,0): 0.5, (2,0): 0.5, (3,0): 0.5,
        (0,1): 2.0, (1,1): 0.5, (2,1): 0.5, (3,1): 0.5,
        (0,2): 0.5, (1,2): 0.5, (2,2): 0.5, (3,2): 0.5
    }

    assert poly_ising.get_spin_scale_factor() == 1

    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.4999, (0,0))
    
    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.5, (10,0))

    with pytest.raises(ValueError):
        PolyIsing(lattice=square, interaction={-1: -1.0})

    with pytest.raises(ValueError):
        PolyIsing(lattice=square, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=1.5, spin_scale_factor=0)


def test_poly_ising_cubic():
    cubic = Cubic(x_size=4, y_size=3, z_size=2, boundary_condition="OBC")
    poly_ising = PolyIsing(lattice=cubic, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=3, spin_scale_factor=4)
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3
    assert poly_ising.get_spin_magnitude() == {
        (0,0,0): 3, (1,0,0): 3, (2,0,0): 3, (3,0,0): 3,
        (0,1,0): 3, (1,1,0): 3, (2,1,0): 3, (3,1,0): 3,
        (0,2,0): 3, (1,2,0): 3, (2,2,0): 3, (3,2,0): 3,
        (0,0,1): 3, (1,0,1): 3, (2,0,1): 3, (3,0,1): 3,
        (0,1,1): 3, (1,1,1): 3, (2,1,1): 3, (3,1,1): 3,
        (0,2,1): 3, (1,2,1): 3, (2,2,1): 3, (3,2,1): 3,
    }
    poly_ising.set_spin_magnitude(2, (0,0,1))
    assert poly_ising.get_spin_magnitude() == {
        (0,0,0): 3, (1,0,0): 3, (2,0,0): 3, (3,0,0): 3,
        (0,1,0): 3, (1,1,0): 3, (2,1,0): 3, (3,1,0): 3,
        (0,2,0): 3, (1,2,0): 3, (2,2,0): 3, (3,2,0): 3,
        (0,0,1): 2, (1,0,1): 3, (2,0,1): 3, (3,0,1): 3,
        (0,1,1): 3, (1,1,1): 3, (2,1,1): 3, (3,1,1): 3,
        (0,2,1): 3, (1,2,1): 3, (2,2,1): 3, (3,2,1): 3,
    }

    assert poly_ising.get_spin_scale_factor() == 4

    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.4999, (0,0,0))
    
    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.5, (10,0,0))

    with pytest.raises(ValueError):
        PolyIsing(lattice=cubic, interaction={-1: -1.0})

    with pytest.raises(ValueError):
        PolyIsing(lattice=cubic, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=1.5, spin_scale_factor=0)


def test_poly_ising_infinite_range():
    infinite_range = InfiniteRange(system_size=4)
    poly_ising = PolyIsing(lattice=infinite_range, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=1, spin_scale_factor=2)
    assert poly_ising.get_interaction() == {0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}
    assert poly_ising.get_degree() == 3
    assert poly_ising.get_spin_magnitude() == {0: 1, 1: 1, 2: 1, 3: 1}
    poly_ising.set_spin_magnitude(2, 0)
    assert poly_ising.get_spin_magnitude() == {0: 2, 1: 1, 2: 1, 3: 1}
    assert poly_ising.get_spin_scale_factor() == 2

    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.4999, 0)
    
    with pytest.raises(ValueError):
        poly_ising.set_spin_magnitude(1.5, 10)

    with pytest.raises(ValueError):
        PolyIsing(lattice=infinite_range, interaction={-1: -1.0})

    with pytest.raises(ValueError):
        PolyIsing(lattice=infinite_range, interaction={0: -1.0, 1: +2.0, 2: -3.0, 3: +4.0}, spin_magnitude=1.5, spin_scale_factor=0)


