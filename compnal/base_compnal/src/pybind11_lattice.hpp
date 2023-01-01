//
//  Copyright 2022 Kohei Suzuki
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
//  pybind11_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/17.
//  
//

#ifndef COMPNAL_PYBIND11_LATTICE_HPP_
#define COMPNAL_PYBIND11_LATTICE_HPP_

#include "../../../cpp_compnal/src/lattice/all.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace compnal {
namespace wrapper {

namespace py = pybind11;

//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

void pybind11BoundaryCondition(py::module &m) {
   py::enum_<compnal::lattice::BoundaryCondition>(m, "BoundaryCondition")
      .value("NONE", compnal::lattice::BoundaryCondition::NONE)
      .value("OBC", compnal::lattice::BoundaryCondition::OBC)
      .value("PBC", compnal::lattice::BoundaryCondition::PBC);
}

void pybind11LatticeChain(py::module &m) {
   
   using ODL = lattice::Chain;
   auto py_class = py::class_<ODL>(m, "Chain", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t, const lattice::BoundaryCondition>(), "system_size"_a, "boundary_condition"_a);
   
   //Public Member Functions
   py_class.def("get_system_size", &ODL::GetSystemSize);
   py_class.def("get_boundary_condition", &ODL::GetBoundaryCondition);
   
}

void pybind11LatticeSquare(py::module &m) {
   
   using TDL = lattice::Square;
   auto py_class = py::class_<TDL>(m, "Square", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t, const std::int32_t, const lattice::BoundaryCondition>(), "x_size"_a, "y_size"_a, "boundary_condition"_a);
   
   //Public Member Functions
   py_class.def("get_x_size", &TDL::GetXSize);
   py_class.def("get_y_size", &TDL::GetYSize);
   py_class.def("get_boundary_condition", &TDL::GetBoundaryCondition);
   py_class.def("get_system_size", &TDL::GetSystemSize);


}


void pybind11LatticeCubic(py::module &m) {
   
   using TDL = lattice::Cubic;
   auto py_class = py::class_<TDL>(m, "Cubic", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t, const std::int32_t, const std::int32_t, const lattice::BoundaryCondition>(),
                "x_size"_a, "y_size"_a, "z_size"_a, "boundary_condition"_a);
   
   //Public Member Functions
   py_class.def("get_x_size", &TDL::GetXSize);
   py_class.def("get_y_size", &TDL::GetYSize);
   py_class.def("get_z_size", &TDL::GetZSize);
   py_class.def("get_system_size", &TDL::GetSystemSize);
   py_class.def("get_boundary_condition", &TDL::GetBoundaryCondition);
   
}

void pybind11LatticeInfiniteRange(py::module &m) {
   
   using TDL = lattice::InfiniteRange;
   auto py_class = py::class_<TDL>(m, "InfiniteRange", py::module_local());
   
   //Constructors
   py_class.def(py::init<const std::int32_t>(), "system_size"_a);
   
   //Public Member Functions
   py_class.def("get_system_size", &TDL::GetSystemSize);
   py_class.def("get_boundary_condition", &TDL::GetBoundaryCondition);
   
}

void pybind11LatticeAnyLattice(py::module &m) {
   
   using TDL = lattice::AnyLattice;
   auto py_class = py::class_<TDL>(m, "AnyLattice", py::module_local());
   
   py_class.def(py::init<>());
   
   //Public Member Functions
   py_class.def("get_boundary_condition", &TDL::GetBoundaryCondition);
   
}

} // namespace wrapper
} // namespace compnal


#endif /* COMPNAL_PYBIND11_LATTICE_HPP_ */
