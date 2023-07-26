import pytest

from compnal.lattice import (
    BoundaryCondition,
    Chain,
    Cubic,
    InfiniteRange,
    LatticeType,
    Square,
)
from compnal.model.classical import ClassicalModelType, Ising


def test_ising_chain():
    chain = Chain(system_size=3, boundary_condition="OBC")
    ising = Ising(lattice=chain, linear=1.0, quadratic=-4.0, spin_magnitude=1)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == -4.0
    assert ising.get_spin_magnitude() == {(0,): 1, (1,): 1, (2,): 1}
    assert ising.get_spin_scale_factor() == 1
    ising.set_spin_magnitude(2, (0,))
    assert ising.get_spin_magnitude() == {(0,): 2, (1,): 1, (2,): 1}

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, (0,))

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, (10,))

    with pytest.raises(ValueError):
        Ising(
            lattice=chain,
            linear=1.0,
            quadratic=-4.0,
            spin_magnitude=1.5,
            spin_scale_factor=0,
        )

    info = ising.export_info()
    assert info.model_type == ClassicalModelType.ISING
    assert info.interactions == {1: 1.0, 2: -4.0}
    assert info.spin_magnitude == {(0,): 2, (1,): 1, (2,): 1}
    assert info.spin_scale_factor == 1
    assert info.lattice.lattice_type == LatticeType.CHAIN
    assert info.lattice.system_size == 3
    assert info.lattice.shape == (3,)
    assert info.lattice.boundary_condition == BoundaryCondition.OBC


def test_ising_chain_serializable():
    chain = Chain(system_size=3, boundary_condition="OBC")
    ising = Ising(lattice=chain, linear=1.0, quadratic=-4.0, spin_magnitude=1)
    obj = ising.to_serializable()
    assert obj["model_type"] == ClassicalModelType.ISING
    assert obj["interactions"] == {1: 1.0, 2: -4.0}
    assert obj["spin_magnitude_values"] == list({(0,): 1, (1,): 1, (2,): 1}.values())
    assert obj["spin_magnitude_keys"] == list({(0,): 1, (1,): 1, (2,): 1}.keys())
    assert obj["spin_scale_factor"] == 1
    assert obj["lattice"]["lattice_type"] == LatticeType.CHAIN
    assert obj["lattice"]["system_size"] == 3
    assert obj["lattice"]["shape"] == (3,)
    assert obj["lattice"]["boundary_condition"] == BoundaryCondition.OBC

    ising = Ising.from_serializable(obj)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == -4.0
    assert ising.get_spin_magnitude() == {(0,): 1, (1,): 1, (2,): 1}
    assert ising.get_spin_scale_factor() == 1


def test_ising_square():
    square = Square(x_size=3, y_size=2, boundary_condition="OBC")
    ising = Ising(lattice=square, linear=1.0, quadratic=2.0)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {
        (0, 0): 0.5,
        (1, 0): 0.5,
        (2, 0): 0.5,
        (0, 1): 0.5,
        (1, 1): 0.5,
        (2, 1): 0.5,
    }
    assert ising.get_spin_scale_factor() == 1
    ising.set_spin_magnitude(2, (0, 1))
    assert ising.get_spin_magnitude() == {
        (0, 0): 0.5,
        (1, 0): 0.5,
        (2, 0): 0.5,
        (0, 1): 2,
        (1, 1): 0.5,
        (2, 1): 0.5,
    }

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, (0, 0))

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, (10, 0))

    with pytest.raises(ValueError):
        Ising(
            lattice=square,
            linear=1.0,
            quadratic=-4.0,
            spin_magnitude=1.5,
            spin_scale_factor=0,
        )

    info = ising.export_info()
    assert info.model_type == ClassicalModelType.ISING
    assert info.interactions == {1: 1.0, 2: 2.0}
    assert info.spin_magnitude == {
        (0, 0): 0.5,
        (1, 0): 0.5,
        (2, 0): 0.5,
        (0, 1): 2,
        (1, 1): 0.5,
        (2, 1): 0.5,
    }
    assert info.spin_scale_factor == 1
    assert info.lattice.lattice_type == LatticeType.SQUARE
    assert info.lattice.system_size == 6
    assert info.lattice.shape == (3, 2)
    assert info.lattice.boundary_condition == BoundaryCondition.OBC


def test_ising_square_serializable():
    square = Square(x_size=3, y_size=2, boundary_condition="OBC")
    ising = Ising(lattice=square, linear=1.0, quadratic=2.0)
    obj = ising.to_serializable()
    assert obj["model_type"] == ClassicalModelType.ISING
    assert obj["interactions"] == {1: 1.0, 2: 2.0}
    assert obj["spin_magnitude_values"] == list(
        {
            (0, 0): 0.5,
            (1, 0): 0.5,
            (2, 0): 0.5,
            (0, 1): 0.5,
            (1, 1): 0.5,
            (2, 1): 0.5,
        }.values()
    )
    assert obj["spin_magnitude_keys"] == list(
        {
            (0, 0): 0.5,
            (1, 0): 0.5,
            (2, 0): 0.5,
            (0, 1): 0.5,
            (1, 1): 0.5,
            (2, 1): 0.5,
        }.keys()
    )
    assert obj["spin_scale_factor"] == 1
    assert obj["lattice"]["lattice_type"] == LatticeType.SQUARE
    assert obj["lattice"]["system_size"] == 6
    assert obj["lattice"]["shape"] == (3, 2)
    assert obj["lattice"]["boundary_condition"] == BoundaryCondition.OBC

    ising = Ising.from_serializable(obj)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {
        (0, 0): 0.5,
        (1, 0): 0.5,
        (2, 0): 0.5,
        (0, 1): 0.5,
        (1, 1): 0.5,
        (2, 1): 0.5,
    }
    assert ising.get_spin_scale_factor() == 1


def test_ising_cubic():
    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="OBC")
    ising = Ising(
        lattice=cubic, linear=1.0, quadratic=2.0, spin_magnitude=3, spin_scale_factor=2
    )
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {
        (0, 0, 0): 3,
        (1, 0, 0): 3,
        (2, 0, 0): 3,
        (0, 1, 0): 3,
        (1, 1, 0): 3,
        (2, 1, 0): 3,
        (0, 0, 1): 3,
        (1, 0, 1): 3,
        (2, 0, 1): 3,
        (0, 1, 1): 3,
        (1, 1, 1): 3,
        (2, 1, 1): 3,
    }
    assert ising.get_spin_scale_factor() == 2
    ising.set_spin_magnitude(2, (0, 0, 1))
    assert ising.get_spin_magnitude() == {
        (0, 0, 0): 3,
        (1, 0, 0): 3,
        (2, 0, 0): 3,
        (0, 1, 0): 3,
        (1, 1, 0): 3,
        (2, 1, 0): 3,
        (0, 0, 1): 2,
        (1, 0, 1): 3,
        (2, 0, 1): 3,
        (0, 1, 1): 3,
        (1, 1, 1): 3,
        (2, 1, 1): 3,
    }

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, (0, 0, 0))

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, (10, 0, 0))

    with pytest.raises(ValueError):
        Ising(
            lattice=cubic,
            linear=1.0,
            quadratic=-4.0,
            spin_magnitude=1.5,
            spin_scale_factor=0,
        )

    info = ising.export_info()
    assert info.model_type == ClassicalModelType.ISING
    assert info.interactions == {1: 1.0, 2: 2.0}
    assert info.spin_magnitude == {
        (0, 0, 0): 3,
        (1, 0, 0): 3,
        (2, 0, 0): 3,
        (0, 1, 0): 3,
        (1, 1, 0): 3,
        (2, 1, 0): 3,
        (0, 0, 1): 2,
        (1, 0, 1): 3,
        (2, 0, 1): 3,
        (0, 1, 1): 3,
        (1, 1, 1): 3,
        (2, 1, 1): 3,
    }
    assert info.spin_scale_factor == 2
    assert info.lattice.lattice_type == LatticeType.CUBIC
    assert info.lattice.system_size == 12
    assert info.lattice.shape == (3, 2, 2)
    assert info.lattice.boundary_condition == BoundaryCondition.OBC


def test_ising_cubic_serializable():
    cubic = Cubic(x_size=3, y_size=2, z_size=2, boundary_condition="OBC")
    ising = Ising(
        lattice=cubic, linear=1.0, quadratic=2.0, spin_magnitude=3, spin_scale_factor=2
    )
    obj = ising.to_serializable()
    assert obj["model_type"] == ClassicalModelType.ISING
    assert obj["interactions"] == {1: 1.0, 2: 2.0}
    assert obj["spin_magnitude_values"] == list(
        {
            (0, 0, 0): 3,
            (1, 0, 0): 3,
            (2, 0, 0): 3,
            (0, 1, 0): 3,
            (1, 1, 0): 3,
            (2, 1, 0): 3,
            (0, 0, 1): 3,
            (1, 0, 1): 3,
            (2, 0, 1): 3,
            (0, 1, 1): 3,
            (1, 1, 1): 3,
            (2, 1, 1): 3,
        }.values()
    )
    assert obj["spin_magnitude_keys"] == list(
        {
            (0, 0, 0): 3,
            (1, 0, 0): 3,
            (2, 0, 0): 3,
            (0, 1, 0): 3,
            (1, 1, 0): 3,
            (2, 1, 0): 3,
            (0, 0, 1): 3,
            (1, 0, 1): 3,
            (2, 0, 1): 3,
            (0, 1, 1): 3,
            (1, 1, 1): 3,
            (2, 1, 1): 3,
        }.keys()
    )
    assert obj["spin_scale_factor"] == 2
    assert obj["lattice"]["lattice_type"] == LatticeType.CUBIC
    assert obj["lattice"]["system_size"] == 12
    assert obj["lattice"]["shape"] == (3, 2, 2)
    assert obj["lattice"]["boundary_condition"] == BoundaryCondition.OBC

    ising = Ising.from_serializable(obj)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {
        (0, 0, 0): 3,
        (1, 0, 0): 3,
        (2, 0, 0): 3,
        (0, 1, 0): 3,
        (1, 1, 0): 3,
        (2, 1, 0): 3,
        (0, 0, 1): 3,
        (1, 0, 1): 3,
        (2, 0, 1): 3,
        (0, 1, 1): 3,
        (1, 1, 1): 3,
        (2, 1, 1): 3,
    }
    assert ising.get_spin_scale_factor() == 2


def test_ising_infinite_range():
    infinite_range = InfiniteRange(system_size=3)
    ising = Ising(
        lattice=infinite_range,
        linear=1.0,
        quadratic=2.0,
        spin_magnitude=1,
        spin_scale_factor=3,
    )
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {(0,): 1, (1,): 1, (2,): 1}
    assert ising.get_spin_scale_factor() == 3
    ising.set_spin_magnitude(2, (0,))
    assert ising.get_spin_magnitude() == {(0,): 2, (1,): 1, (2,): 1}

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.4999, (0,))

    with pytest.raises(ValueError):
        ising.set_spin_magnitude(1.5, (10,))

    with pytest.raises(ValueError):
        Ising(
            lattice=infinite_range,
            linear=1.0,
            quadratic=-4.0,
            spin_magnitude=1.5,
            spin_scale_factor=0,
        )

    info = ising.export_info()
    assert info.model_type == ClassicalModelType.ISING
    assert info.interactions == {1: 1.0, 2: 2.0}
    assert info.spin_magnitude == {(0,): 2, (1,): 1, (2,): 1}
    assert info.spin_scale_factor == 3
    assert info.lattice.lattice_type == LatticeType.INFINITE_RANGE
    assert info.lattice.system_size == 3
    assert info.lattice.shape == None
    assert info.lattice.boundary_condition == BoundaryCondition.NONE


def test_ising_infinite_range_serializable():
    infinite_range = InfiniteRange(system_size=3)
    ising = Ising(
        lattice=infinite_range,
        linear=1.0,
        quadratic=2.0,
        spin_magnitude=1,
        spin_scale_factor=3,
    )
    obj = ising.to_serializable()
    assert obj["model_type"] == ClassicalModelType.ISING
    assert obj["interactions"] == {1: 1.0, 2: 2.0}
    assert obj["spin_magnitude_values"] == list({(0,): 1, (1,): 1, (2,): 1}.values())
    assert obj["spin_magnitude_keys"] == list({(0,): 1, (1,): 1, (2,): 1}.keys())
    assert obj["spin_scale_factor"] == 3
    assert obj["lattice"]["lattice_type"] == LatticeType.INFINITE_RANGE
    assert obj["lattice"]["system_size"] == 3
    assert obj["lattice"]["shape"] == None
    assert obj["lattice"]["boundary_condition"] == BoundaryCondition.NONE

    ising = Ising.from_serializable(obj)
    assert ising.get_linear() == 1.0
    assert ising.get_quadratic() == 2.0
    assert ising.get_spin_magnitude() == {(0,): 1, (1,): 1, (2,): 1}
    assert ising.get_spin_scale_factor() == 3
