from compnal import lattice, model
import math

def test_model_polynomial_ising_chain():
    ising = model.PolynomialIsing(
        lattice=lattice.Chain(system_size=4, boundary_condition=lattice.BoundaryCondition.PBC), interaction={1: 2.0, 3: 7.1}
    )
    assert ising.get_interaction() == {1: 2.0, 3: 7.1}
    assert ising.get_system_size() == 4
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.PBC
    assert ising.get_degree() == 3
    assert math.isclose(ising.calculate_energy([-1, +1, +1, -1]), 0)
    assert ising.system_size == 4
    assert ising.boundary_condition == lattice.BoundaryCondition.PBC

def test_model_polynomial_ising_square():

    ising = model.PolynomialIsing(
        lattice=lattice.Square(x_size=2, y_size=3, boundary_condition=lattice.BoundaryCondition.PBC), interaction={1: 2.0, 3: 7.1}
    )
    assert ising.get_interaction() == {1: 2.0, 3: 7.1}
    assert ising.get_system_size() == 6
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.PBC
    assert ising.get_degree() == 3
    assert math.isclose(ising.calculate_energy([-1, +1, +1, -1, -1, -1]), 0)
    assert ising.system_size == 6
    assert ising.boundary_condition == lattice.BoundaryCondition.PBC

def test_model_polynomial_ising_cubic():
    ising = model.PolynomialIsing(
        lattice=lattice.Cubic(x_size=2, y_size=2, z_size=2, boundary_condition=lattice.BoundaryCondition.PBC),
        interaction={1: 2.0, 3: 7.1},
    )
    assert ising.get_interaction() == {1: 2.0, 3: 7.1}
    assert ising.get_system_size() == 8
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.PBC
    assert ising.get_degree() == 3
    assert math.isclose(ising.calculate_energy([-1, +1, +1, -1, -1, -1, +1, +1]), 0)
    assert ising.system_size == 8
    assert ising.boundary_condition == lattice.BoundaryCondition.PBC

def test_model_polynomial_ising_infinite_range():
    ising = model.PolynomialIsing(
        lattice=lattice.InfiniteRange(system_size=4), interaction={1: 2.0, 3: 7.1}
    )
    assert ising.get_interaction() == {1: 2.0, 3: 7.1}
    assert ising.get_system_size() == 4
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.NONE
    assert ising.get_degree() == 3
    assert math.isclose(ising.calculate_energy([+1, +1, -1, -1]), 0)
    assert ising.system_size == 4
    assert ising.boundary_condition == lattice.BoundaryCondition.NONE

def test_model_polynomial_ising_any_lattice():
    ising = model.PolynomialIsing(
        lattice=lattice.AnyLattice(),
        interaction={(1,): -1, ("a", 1): +2, ("b",): +1, ("a", "b", 1): +3}
    )
    assert ising.get_interaction() == {(1,): -1, ("a", 1): +2, ("b",): +1, ("a", "b", 1): +3}
    assert ising.get_system_size() == 3
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.NONE
    assert ising.get_degree() == 3
    assert math.isclose(ising.calculate_energy([+1, +1, -1]), -1)
    assert ising.system_size == 3
    assert ising.boundary_condition == lattice.BoundaryCondition.NONE
