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


from compnal.lattice.boundary_condition import BoundaryCondition, _cast_base_boundary_condition, _cast_boundary_condition
from compnal.base_compnal import base_lattice

def test_cast_base_boundary_condition():
    assert _cast_base_boundary_condition(base_lattice.BoundaryCondition.NONE) == BoundaryCondition.NONE
    assert _cast_base_boundary_condition(base_lattice.BoundaryCondition.OBC) == BoundaryCondition.OBC
    assert _cast_base_boundary_condition(base_lattice.BoundaryCondition.PBC) == BoundaryCondition.PBC

def test_cast_boundary_condition():
    assert _cast_boundary_condition("NONE") == base_lattice.BoundaryCondition.NONE
    assert _cast_boundary_condition("OBC") == base_lattice.BoundaryCondition.OBC
    assert _cast_boundary_condition("PBC") == base_lattice.BoundaryCondition.PBC

    assert _cast_boundary_condition(BoundaryCondition.NONE) == base_lattice.BoundaryCondition.NONE
    assert _cast_boundary_condition(BoundaryCondition.OBC) == base_lattice.BoundaryCondition.OBC
    assert _cast_boundary_condition(BoundaryCondition.PBC) == base_lattice.BoundaryCondition.PBC

    assert _cast_boundary_condition(None) == base_lattice.BoundaryCondition.NONE