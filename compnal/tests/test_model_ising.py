from compnal import lattice, model

def test_model_ising_basic():
    ising = model.make_ising(
        lattice=lattice.Chain(system_size=10), linear=1.1, quadratic=-2.3
    )
    ising.set_constant(3.0)
    assert ising.get_interaction() == (3.0, 1.1, -2.3)
    assert ising.get_system_size() == 10
    assert ising.system_size == 10
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.OBC
    assert ising.boundary_condition == lattice.BoundaryCondition.OBC
    assert ising.get_degree() == 2

    ising = model.make_ising(
        lattice=lattice.Square(x_size=3, y_size=4), linear=1.1, quadratic=-2.3
    )
    ising.set_constant(3.0)
    assert ising.get_interaction() == (3.0, 1.1, -2.3)
    assert ising.get_system_size() == 12
    assert ising.system_size == 12
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.OBC
    assert ising.boundary_condition == lattice.BoundaryCondition.OBC
    assert ising.get_degree() == 2

    ising = model.make_ising(
        lattice=lattice.Cubic(x_size=3, y_size=4, z_size=2), linear=1.1, quadratic=-2.3
    )
    ising.set_constant(3.0)
    assert ising.get_interaction() == (3.0, 1.1, -2.3)
    assert ising.get_system_size() == 24
    assert ising.system_size == 24
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.OBC
    assert ising.boundary_condition == lattice.BoundaryCondition.OBC
    assert ising.get_degree() == 2

    ising = model.make_ising(
        lattice=lattice.InfiniteRange(system_size=10), linear=1.1, quadratic=-2.3
    )
    ising.set_constant(3.0)
    assert ising.get_interaction() == (3.0, 1.1, -2.3)
    assert ising.get_system_size() == 10
    assert ising.system_size == 10
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.NONE
    assert ising.boundary_condition == lattice.BoundaryCondition.NONE
    assert ising.get_degree() == 2

    ising = model.make_ising(
        lattice=lattice.AnyLattice(),
        linear={1: 1.0, "a": 1.2, ("a", 3): 3},
        quadratic={(1, "a"): -2.0, (1, (2, "a")): -3.1},
    )
    ising.set_constant(3.0)
    assert ising.get_interaction() == {
        (): 3.0,
        (1,): 1.0,
        ("a",): 1.2,
        (("a", 3),): 3,
        (1, "a"): -2.0,
        (1, (2, "a")): -3.1,
    }
    assert ising.get_system_size() == 4
    assert ising.system_size == 4
    assert ising.get_boundary_condition() == lattice.BoundaryCondition.NONE
    assert ising.boundary_condition == lattice.BoundaryCondition.NONE
    assert ising.get_degree() == 2
    assert sorted(ising.generate_index_list(), key=lambda x: str(x)) == sorted([1, "a", ["a", 3], [2, "a"]], key=lambda x: str(x))
