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
//  Created by Kohei Suzuki on 2022/07/13.
//

#include "src/pybind11_lattice.hpp"
#include "src/pybind11_model.hpp"
#include "src/pybind11_solver.hpp"


PYBIND11_MODULE(base_compnal, m) {
   namespace py = pybind11;
   
   using RealType = double;
      
   py::module_ m_lattice = m.def_submodule("base_lattice");
   compnal::wrapper::pybind11BoundaryCondition(m_lattice);
   compnal::wrapper::pybind11LatticeChain(m_lattice);
   compnal::wrapper::pybind11LatticeSquare(m_lattice);
   compnal::wrapper::pybind11LatticeCubic(m_lattice);
   compnal::wrapper::pybind11LatticeInfiniteRange(m_lattice);
   compnal::wrapper::pybind11LatticeAnyLattice(m_lattice);

   py::module_ m_model = m.def_submodule("base_model");
   compnal::wrapper::pybind11ModelIsing<compnal::lattice::Chain, RealType>(m_model, "Chain");
   compnal::wrapper::pybind11ModelIsing<compnal::lattice::Square, RealType>(m_model, "Square");
   compnal::wrapper::pybind11ModelIsing<compnal::lattice::Cubic, RealType>(m_model, "Cubic");
   compnal::wrapper::pybind11ModelIsing<compnal::lattice::InfiniteRange, RealType>(m_model, "InfiniteRange");
   compnal::wrapper::pybind11ModelIsing<compnal::lattice::AnyLattice, RealType>(m_model, "AnyLattice");

   compnal::wrapper::pybind11ModelPolynomialIsing<compnal::lattice::Chain, RealType>(m_model, "Chain");
   compnal::wrapper::pybind11ModelPolynomialIsing<compnal::lattice::Square, RealType>(m_model, "Square");
   compnal::wrapper::pybind11ModelPolynomialIsing<compnal::lattice::Cubic, RealType>(m_model, "Cubic");
   compnal::wrapper::pybind11ModelPolynomialIsing<compnal::lattice::InfiniteRange, RealType>(m_model, "InfiniteRange");
   compnal::wrapper::pybind11ModelPolynomialIsing<compnal::lattice::AnyLattice, RealType>(m_model, "AnyLattice");
   
   py::module_ m_solver = m.def_submodule("base_solver");
   compnal::wrapper::pybind11SolverCMCUpdater(m_solver);
   
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::PolynomialIsing<compnal::lattice::Chain, RealType>>(m_solver, "PolyIsingChain");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::PolynomialIsing<compnal::lattice::Square, RealType>>(m_solver, "PolyIsingSquare");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::PolynomialIsing<compnal::lattice::Cubic, RealType>>(m_solver, "PolyIsingCubic");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::PolynomialIsing<compnal::lattice::InfiniteRange, RealType>>(m_solver, "PolyIsingInfiniteRange");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::PolynomialIsing<compnal::lattice::AnyLattice, RealType>>(m_solver, "PolyIsingAnyLattice");
   
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::Ising<compnal::lattice::Chain, RealType>>(m_solver, "IsingChain");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::Ising<compnal::lattice::Square, RealType>>(m_solver, "IsingSquare");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::Ising<compnal::lattice::Cubic, RealType>>(m_solver, "IsingCubic");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::Ising<compnal::lattice::InfiniteRange, RealType>>(m_solver, "IsingInfiniteRange");
   compnal::wrapper::pybind11SolverClassicalMonteCarlo<compnal::model::Ising<compnal::lattice::AnyLattice, RealType>>(m_solver, "IsingAnyLattice");

};
