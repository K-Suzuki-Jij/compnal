#  Copyright 2023 Kohei Suzuki
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import pytest
from compnal.lattice import Chain, BoundaryCondition
from compnal.lattice import LatticeType

def test_chain():
    chain = Chain(system_size=3, boundary_condition="OBC")
    assert chain.system_size == 3
    assert chain.boundary_condition == BoundaryCondition.OBC
    assert chain.generate_coordinate_list() == [(0,), (1,), (2,)]

    with pytest.raises(ValueError):
        Chain(system_size=3, boundary_condition="ABC")
    
    with pytest.raises(ValueError):
        Chain(system_size=0, boundary_condition="OBC")

    info = chain.export_info()
    assert info.lattice_type == LatticeType.CHAIN
    assert info.system_size == 3
    assert info.shape == (3,)
    assert info.boundary_condition == BoundaryCondition.OBC

def test_chain_serializable():
    chain = Chain(system_size=3, boundary_condition="OBC")
    obj = chain.to_serializable()
    assert obj["lattice_type"] == LatticeType.CHAIN
    assert obj["system_size"] == 3
    assert obj["shape"] == (3,)
    assert obj["boundary_condition"] == BoundaryCondition.OBC

    chain = Chain.from_serializable(obj)
    assert chain.system_size == 3
    assert chain.boundary_condition == BoundaryCondition.OBC
    assert chain.generate_coordinate_list() == [(0,), (1,), (2,)]