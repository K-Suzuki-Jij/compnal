import pytest
from compnal.model.classical import ClassicalModelType, ClassicalModelInfo
from compnal.lattice import LatticeInfo, LatticeType, BoundaryCondition


def test_classical_model_info():
    info = ClassicalModelInfo.from_serializable(
            obj=ClassicalModelInfo(
                model_type=ClassicalModelType.ISING,
                interactions={0: 1.0, 1: 2.0},
                spin_magnitude={(0, 0): 1.0, (0, 1): 2.0},
                spin_scale_factor=1.0,
                lattice=LatticeInfo(
                    lattice_type=LatticeType.SQUARE,
                    system_size=4,
                    shape=(2, 2),
                    boundary_condition=BoundaryCondition.OBC,
                )
            ).to_serializable()
    )

    assert info.model_type == ClassicalModelType.ISING
    assert info.interactions == {0: 1.0, 1: 2.0}
    assert info.spin_magnitude == {(0, 0): 1.0, (0, 1): 2.0}
    assert info.spin_scale_factor == 1.0
    assert info.lattice.lattice_type == LatticeType.SQUARE
    assert info.lattice.system_size == 4
    assert info.lattice.shape == (2, 2)
    assert info.lattice.boundary_condition == BoundaryCondition.OBC