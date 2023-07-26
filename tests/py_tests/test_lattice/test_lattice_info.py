from compnal.lattice import BoundaryCondition, LatticeInfo, LatticeType


def test_lattice_info():
    lattice_info = LatticeInfo.from_serializable(
        LatticeInfo(
            lattice_type=LatticeType.SQUARE,
            system_size=4,
            shape=(2, 2),
            boundary_condition=BoundaryCondition.PBC,
        ).to_serializable()
    )

    assert lattice_info.lattice_type == LatticeType.SQUARE
    assert lattice_info.system_size == 4
    assert lattice_info.shape == (2, 2)
    assert lattice_info.boundary_condition == BoundaryCondition.PBC
