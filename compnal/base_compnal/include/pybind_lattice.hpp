//
//  Copyright 2023 Kohei Suzuki
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  pybind_lattice.hpp
//  compnal
//
//  Created by kohei on 2023/04/30.
//  
//

#pragma once

#include "../../../include/lattice/all.hpp"

namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

void PyBindBoundaryCondition(py::module &m) {
   py::enum_<compnal::lattice::BoundaryCondition>(m, "BoundaryCondition")
      .value("NONE", compnal::lattice::BoundaryCondition::NONE)
      .value("OBC", compnal::lattice::BoundaryCondition::OBC)
      .value("PBC", compnal::lattice::BoundaryCondition::PBC);
}

void PyBindLatticeChain(py::module &m) {
   
   using LAT = lattice::Chain;
   auto py_class = py::class_<LAT>(m, "Chain", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t, const lattice::BoundaryCondition>(), "system_size"_a, "boundary_condition"_a);
   
   //Public Member Functions
   py_class.def("get_system_size", &LAT::GetSystemSize);
   py_class.def("get_boundary_condition", &LAT::GetBoundaryCondition);
   py_class.def("generate_coordinate_list", &LAT::GenerateCoordinateList);
   
}

void PyBindLatticeSquare(py::module &m) {
   
   using LAT = lattice::Square;
   auto py_class = py::class_<LAT>(m, "Square", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t, const std::int32_t, const lattice::BoundaryCondition>(), "x_size"_a, "y_size"_a, "boundary_condition"_a);
   
   //Public Member Functions
   py_class.def("get_x_size", &LAT::GetXSize);
   py_class.def("get_y_size", &LAT::GetYSize);
   py_class.def("get_boundary_condition", &LAT::GetBoundaryCondition);
   py_class.def("get_system_size", &LAT::GetSystemSize);
   py_class.def("generate_coordinate_list", &LAT::GenerateCoordinateList);

}

void PyBindLatticeCubic(py::module &m) {
   
   using LAT = lattice::Cubic;
   auto py_class = py::class_<LAT>(m, "Cubic", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t, const std::int32_t, const std::int32_t, const lattice::BoundaryCondition>(),
                "x_size"_a, "y_size"_a, "z_size"_a, "boundary_condition"_a);
   
   //Public Member Functions
   py_class.def("get_x_size", &LAT::GetXSize);
   py_class.def("get_y_size", &LAT::GetYSize);
   py_class.def("get_z_size", &LAT::GetZSize);
   py_class.def("get_system_size", &LAT::GetSystemSize);
   py_class.def("get_boundary_condition", &LAT::GetBoundaryCondition);
   py_class.def("generate_coordinate_list", &LAT::GenerateCoordinateList);
   
}

void PyBindLatticeInfiniteRange(py::module &m) {
   
   using LAT = lattice::InfiniteRange;
   auto py_class = py::class_<LAT>(m, "InfiniteRange", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t>(), "system_size"_a);
   
   //Public Member Functions
   py_class.def("get_system_size", &LAT::GetSystemSize);
   py_class.def("get_boundary_condition", &LAT::GetBoundaryCondition);
   py_class.def("generate_coordinate_list", &LAT::GenerateCoordinateList);
   
}


} // namespace wrapper
} // namespace compnal
