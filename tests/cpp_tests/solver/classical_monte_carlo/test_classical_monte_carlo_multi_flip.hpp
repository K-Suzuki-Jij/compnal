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
//  test_classical_monte_carlo_multi_flip.hpp
//  compnal
//
//  Created by kohei on 2023/11/07.
//  
//

#pragma once

#include "../../../../include/lattice/all.hpp"
#include "../../../../include/model/classical/ising.hpp"
#include "../../../../include/model/classical/polynomial_ising.hpp"
#include "../../../../include/solver/classical_monte_carlo/classical_monte_carlo.hpp"


namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloMF, PolyIsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using PolyIsing = model::classical::PolynomialIsing<Square>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Square{7, 6, bc}, {{3, -1.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunMultiFlip(poly_ising, 2, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunMultiFlip(poly_ising, 2, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunMultiFlip(poly_ising, 2, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunMultiFlip(poly_ising, 2, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
  
   }
   
}


}  // namespace test
}  // namespace compnal
