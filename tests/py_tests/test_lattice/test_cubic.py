#  Copyright 2024 Kohei Suzuki
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

from compnal.lattice import BoundaryCondition, Cubic, LatticeType


def test_cubic():
    cubic = Cubic(x_size=2, y_size=3, z_size=2, boundary_condition="OBC")
    assert cubic.x_size == 2
    assert cubic.y_size == 3
    assert cubic.z_size == 2
    assert cubic.system_size == 12
    assert cubic.boundary_condition == BoundaryCondition.OBC
    assert cubic.generate_coordinate_list() == [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 2, 0),
        (1, 2, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (0, 2, 1),
        (1, 2, 1),
    ]

    with pytest.raises(ValueError):
        Cubic(x_size=2, y_size=3, z_size=2, boundary_condition="ABC")

    with pytest.raises(ValueError):
        Cubic(x_size=0, y_size=2, z_size=2, boundary_condition="OBC")

    with pytest.raises(ValueError):
        Cubic(x_size=3, y_size=-1, z_size=2, boundary_condition="OBC")

    with pytest.raises(ValueError):
        Cubic(x_size=3, y_size=2, z_size=-1, boundary_condition="OBC")

    info = cubic.export_info()
    assert info.lattice_type == LatticeType.CUBIC
    assert info.system_size == 12
    assert info.shape == (2, 3, 2)
    assert info.boundary_condition == BoundaryCondition.OBC


def test_cubic_serializable():
    cubic = Cubic(x_size=2, y_size=3, z_size=2, boundary_condition="OBC")
    obj = cubic.to_serializable()
    assert obj["lattice_type"] == LatticeType.CUBIC
    assert obj["system_size"] == 12
    assert obj["shape"] == (2, 3, 2)
    assert obj["boundary_condition"] == BoundaryCondition.OBC

    cubic = Cubic.from_serializable(obj)
    assert cubic.x_size == 2
    assert cubic.y_size == 3
    assert cubic.z_size == 2
    assert cubic.system_size == 12
    assert cubic.boundary_condition == BoundaryCondition.OBC
    assert cubic.generate_coordinate_list() == [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 2, 0),
        (1, 2, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (0, 2, 1),
        (1, 2, 1),
    ]
