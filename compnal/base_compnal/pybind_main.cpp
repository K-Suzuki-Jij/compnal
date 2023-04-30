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
//  pybind_main.cpp
//  compnal
//
//  Created by kohei on 2023/04/30.
//  
//

#include "include/pybind_lattice.hpp"

PYBIND11_MODULE(base_compnal, m) {
   namespace py = pybind11;
   
   py::module_ m_lattice = m.def_submodule("base_lattice");
   compnal::wrapper::PyBindBoundaryCondition(m_lattice);
   compnal::wrapper::PyBindLatticeChain(m_lattice);
   compnal::wrapper::PyBindLatticeSquare(m_lattice);
   compnal::wrapper::PyBindLatticeCubic(m_lattice);
   compnal::wrapper::PyBindLatticeInfiniteRange(m_lattice);

};
