from compnal import lattice


def test_lattice_one_dim():
    chain = lattice.Chain(
        system_size=10, boundary_condition=lattice.BoundaryCondition.OBC
    )
    chain.system_size = 7
    chain.boundary_condition = lattice.BoundaryCondition.PBC
    assert chain.get_system_size() == 7
    assert chain.system_size == 7
    assert chain.get_boundary_condition() == lattice.BoundaryCondition.PBC
    assert chain.boundary_condition == lattice.BoundaryCondition.PBC


def test_lattice_two_dim():
    def test_func(LatticeClass) -> None:
        two_dim_lattice = LatticeClass(
            x_size=3, y_size=4, boundary_condition=lattice.BoundaryCondition.PBC
        )
        two_dim_lattice.x_size = 2
        two_dim_lattice.y_size = 3
        two_dim_lattice.boundary_condition = lattice.BoundaryCondition.OBC
        assert two_dim_lattice.get_x_size() == 2
        assert two_dim_lattice.x_size == 2
        assert two_dim_lattice.get_y_size() == 3
        assert two_dim_lattice.y_size == 3
        assert two_dim_lattice.get_system_size() == 6
        assert two_dim_lattice.system_size == 6
        assert two_dim_lattice.get_boundary_condition() == lattice.BoundaryCondition.OBC
        assert two_dim_lattice.boundary_condition == lattice.BoundaryCondition.OBC

    test_func(lattice.Square)


def test_lattice_three_dim():
    cubic = lattice.Cubic(
        x_size=3, y_size=4, z_size=5, boundary_condition=lattice.BoundaryCondition.PBC
    )
    cubic.x_size = 2
    cubic.y_size = 3
    cubic.z_size = 4
    cubic.boundary_condition = lattice.BoundaryCondition.OBC
    assert cubic.get_x_size() == 2
    assert cubic.x_size == 2
    assert cubic.get_y_size() == 3
    assert cubic.y_size == 3
    assert cubic.get_z_size() == 4
    assert cubic.z_size == 4
    assert cubic.get_system_size() == 24
    assert cubic.system_size == 24
    assert cubic.get_boundary_condition() == lattice.BoundaryCondition.OBC
    assert cubic.boundary_condition == lattice.BoundaryCondition.OBC


def test_higher_dim():
    infinite_range = lattice.InfiniteRange(system_size=3)
    infinite_range.system_size = 4
    assert infinite_range.get_boundary_condition() == lattice.BoundaryCondition.NONE
    assert infinite_range.boundary_condition == lattice.BoundaryCondition.NONE
    assert infinite_range.get_system_size() == 4
    assert infinite_range.system_size == 4


def test_any_lattice():
    any_lattice = lattice.AnyLattice()
    assert any_lattice.get_boundary_condition() == lattice.BoundaryCondition.NONE
    assert any_lattice.boundary_condition == lattice.BoundaryCondition.NONE
