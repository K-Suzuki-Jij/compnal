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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>

#include "include/pybind_lattice.hpp"
#include "include/pybind_classical_model.hpp"
#include "include/pybind_classical_monte_carlo.hpp"
#include "include/pybind_solver_parameters.hpp"

PYBIND11_MODULE(base_compnal, m) {
   namespace py = pybind11;
   
   // Lattice
   py::module_ m_lattice = m.def_submodule("base_lattice");
   compnal::wrapper::PyBindBoundaryCondition(m_lattice);
   compnal::wrapper::PyBindLatticeChain(m_lattice);
   compnal::wrapper::PyBindLatticeSquare(m_lattice);
   compnal::wrapper::PyBindLatticeCubic(m_lattice);
   compnal::wrapper::PyBindLatticeInfiniteRange(m_lattice);
   
   // Classical models
   py::module_ m_c_model = m.def_submodule("base_classical_model");
   compnal::wrapper::PyBindClassicalIsing<compnal::lattice::Chain>(m_c_model, "Chain");
   compnal::wrapper::PyBindClassicalIsing<compnal::lattice::Square>(m_c_model, "Square");
   compnal::wrapper::PyBindClassicalIsing<compnal::lattice::Cubic>(m_c_model, "Cubic");
   compnal::wrapper::PyBindClassicalIsing<compnal::lattice::InfiniteRange>(m_c_model, "InfiniteRange");
   compnal::wrapper::PyBindClassicalPolynomialIsing<compnal::lattice::Chain>(m_c_model, "Chain");
   compnal::wrapper::PyBindClassicalPolynomialIsing<compnal::lattice::Square>(m_c_model, "Square");
   compnal::wrapper::PyBindClassicalPolynomialIsing<compnal::lattice::Cubic>(m_c_model, "Cubic");
   compnal::wrapper::PyBindClassicalPolynomialIsing<compnal::lattice::InfiniteRange>(m_c_model, "InfiniteRange");
   
   // Classical Monte Carlo
   py::module_ m_solver = m.def_submodule("base_solver");
   compnal::wrapper::PyBindStateUpdateMethod(m_solver);
   compnal::wrapper::PyBindRandomNumberEngine(m_solver);
   compnal::wrapper::SpinSelectionMethod(m_solver);

   compnal::wrapper::PyBindClassicalMonteCarlo<compnal::model::classical::Ising<compnal::lattice::Chain>>(m_solver, "IsingChain");
   compnal::wrapper::PyBindClassicalMonteCarlo<compnal::model::classical::Ising<compnal::lattice::Square>>(m_solver, "IsingSquare");
   compnal::wrapper::PyBindClassicalMonteCarlo<compnal::model::classical::Ising<compnal::lattice::Cubic>>(m_solver, "IsingCubic");
   compnal::wrapper::PyBindClassicalMonteCarlo<compnal::model::classical::Ising<compnal::lattice::InfiniteRange>>(m_solver, "IsingInfiniteRange");

   compnal::wrapper::PyBindClassicalMonteCarlo<compnal::model::classical::PolynomialIsing<compnal::lattice::Chain>>(m_solver, "PolyIsingChain");
   compnal::wrapper::PyBindClassicalMonteCarlo<compnal::model::classical::PolynomialIsing<compnal::lattice::Square>>(m_solver, "PolyIsingSquare");

};
