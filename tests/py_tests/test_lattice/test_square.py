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

from compnal.lattice import BoundaryCondition, LatticeType, Square


def test_square():
    square = Square(x_size=2, y_size=3, boundary_condition="OBC")
    assert square.x_size == 2
    assert square.y_size == 3
    assert square.system_size == 6
    assert square.boundary_condition == BoundaryCondition.OBC
    assert square.generate_coordinate_list() == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]

    with pytest.raises(ValueError):
        Square(x_size=2, y_size=3, boundary_condition="ABC")

    with pytest.raises(ValueError):
        Square(x_size=0, y_size=2, boundary_condition="OBC")

    with pytest.raises(ValueError):
        Square(x_size=3, y_size=-1, boundary_condition="OBC")

    info = square.export_info()
    assert info.lattice_type == LatticeType.SQUARE
    assert info.system_size == 6
    assert info.shape == (2, 3)
    assert info.boundary_condition == BoundaryCondition.OBC


def test_square_serializable():
    square = Square(x_size=2, y_size=3, boundary_condition="OBC")
    obj = square.to_serializable()
    assert obj["lattice_type"] == LatticeType.SQUARE
    assert obj["system_size"] == 6
    assert obj["shape"] == (2, 3)
    assert obj["boundary_condition"] == BoundaryCondition.OBC

    square = Square.from_serializable(obj)
    assert square.x_size == 2
    assert square.y_size == 3
    assert square.system_size == 6
    assert square.boundary_condition == BoundaryCondition.OBC
    assert square.generate_coordinate_list() == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]
