import pytest
from compnal.lattice.chain import Chain
from compnal.lattice.boundary_condition import BoundaryCondition

def test_chain():
    chain = Chain(system_size=3, boundary_condition="OBC")
    assert chain.system_size == 3
    assert chain.boundary_condition == BoundaryCondition.OBC
    assert chain.generate_coordinate_list() == [0, 1, 2]

    with pytest.raises(ValueError):
        Chain(system_size=3, boundary_condition="ABC")
    
    with pytest.raises(ValueError):
        Chain(system_size=0, boundary_condition="OBC")