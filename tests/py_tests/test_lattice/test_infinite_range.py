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

from compnal.lattice import BoundaryCondition, InfiniteRange, LatticeType


def test_infinite_range():
    infinite_range = InfiniteRange(system_size=3)
    assert infinite_range.system_size == 3
    assert infinite_range.boundary_condition == BoundaryCondition.NONE
    assert infinite_range.generate_coordinate_list() == [(0,), (1,), (2,)]

    with pytest.raises(ValueError):
        InfiniteRange(system_size=0)

    info = infinite_range.export_info()
    assert info.lattice_type == LatticeType.INFINITE_RANGE
    assert info.system_size == 3
    assert info.shape == None
    assert info.boundary_condition == BoundaryCondition.NONE


def test_infinite_serializable():
    infinite_range = InfiniteRange(system_size=3)
    obj = infinite_range.to_serializable()
    assert obj["lattice_type"] == LatticeType.INFINITE_RANGE
    assert obj["system_size"] == 3
    assert obj["shape"] == None
    assert obj["boundary_condition"] == BoundaryCondition.NONE

    infinite_range = InfiniteRange.from_serializable(obj)
    assert infinite_range.system_size == 3
    assert infinite_range.boundary_condition == BoundaryCondition.NONE
    assert infinite_range.generate_coordinate_list() == [(0,), (1,), (2,)]
