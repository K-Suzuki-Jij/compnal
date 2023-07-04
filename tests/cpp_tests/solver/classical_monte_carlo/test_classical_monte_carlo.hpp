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
//  test_classical_monte_carlo.hpp
//  compnal
//
//  Created by kohei on 2023/05/04.
//  
//

#ifndef OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_HPP_
#define OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_HPP_

#include "../../../../include/lattice/all.hpp"
#include "../../../../include/model/classical/ising.hpp"
#include "../../../../include/solver/classical_monte_carlo/classical_monte_carlo.hpp"

namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarlo, SolverParameters) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using Ising = model::classical::Ising<Chain>;
   
   Chain chain{3, BC::OBC};
   Ising ising{chain, 1.0, -4.0};
   
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   const auto samples = cmc.RunSingleFlip(ising, 1000, 4, 2, 1.0, 0,
                                          solver::StateUpdateMethod::METROPOLIS,
                                          solver::RandomNumberEngine::MT,
                                          solver::SpinSelectionMethod::RANDOM);
   EXPECT_EQ(samples.rows(), 4);
   EXPECT_EQ(cmc.CalculateEnergies(ising, samples, 2).size(), 4);
   
   EXPECT_THROW(cmc.RunSingleFlip(ising, -1, 4, 2, 1.0, 0,
                                  solver::StateUpdateMethod::METROPOLIS,
                                  solver::RandomNumberEngine::MT,
                                  solver::SpinSelectionMethod::RANDOM), std::invalid_argument);
   EXPECT_THROW(cmc.RunSingleFlip(ising, 1000, 0, 2, 1.0, 0,
                                  solver::StateUpdateMethod::METROPOLIS,
                                  solver::RandomNumberEngine::MT,
                                  solver::SpinSelectionMethod::RANDOM), std::invalid_argument);
   EXPECT_THROW(cmc.RunSingleFlip(ising, 1000, 4, 0, 1.0, 0,
                                  solver::StateUpdateMethod::METROPOLIS,
                                  solver::RandomNumberEngine::MT,
                                  solver::SpinSelectionMethod::RANDOM), std::invalid_argument);
   EXPECT_THROW(cmc.RunSingleFlip(ising, 1000, 4, 2, -0.9, 0,
                                  solver::StateUpdateMethod::METROPOLIS,
                                  solver::RandomNumberEngine::MT,
                                  solver::SpinSelectionMethod::RANDOM), std::invalid_argument);
   
}

TEST(SolverClassicalMonteCarlo, IsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using Ising = model::classical::Ising<Chain>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      const Ising ising{Chain{3, bc}, -1.0, -4.0, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::SEQUENTIAL));
   }
}

TEST(SolverClassicalMonteCarlo, IsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::Ising<Square>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Square{3, 4, bc}, -1.0, -4.0, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::SEQUENTIAL));
   }
}

TEST(SolverClassicalMonteCarlo, IsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using Ising = model::classical::Ising<Cubic>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Cubic{3, 4, 2, bc}, -1.0, -4.0, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::SEQUENTIAL));
   }
}

TEST(SolverClassicalMonteCarlo, IsingOnInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using InfiniteRange = lattice::InfiniteRange;
   using Ising = model::classical::Ising<InfiniteRange>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   Ising ising{InfiniteRange{3}, -1.0, -4.0, 1.5, 2};
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::METROPOLIS,
                                     solver::RandomNumberEngine::MT,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::MT,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::XORSHIFT,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::MT_64,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(ising, 500, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::MT_64,
                                     solver::SpinSelectionMethod::SEQUENTIAL));
}


TEST(SolverClassicalMonteCarlo, PolyIsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using PolyIsing = model::classical::PolynomialIsing<Chain>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Chain{7, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::SEQUENTIAL));
   }
}

TEST(SolverClassicalMonteCarlo, PolyIsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using PolyIsing = model::classical::PolynomialIsing<Square>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Square{7, 6, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 500, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::SEQUENTIAL));
   }
}

TEST(SolverClassicalMonteCarlo, PolyIsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using PolyIsing = model::classical::PolynomialIsing<Cubic>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Cubic{7, 6, 5, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::METROPOLIS,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::XORSHIFT,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                        solver::StateUpdateMethod::HEAT_BATH,
                                        solver::RandomNumberEngine::MT_64,
                                        solver::SpinSelectionMethod::SEQUENTIAL));
   }
}

TEST(SolverClassicalMonteCarlo, PolyIsingOnInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using InfiniteRange = lattice::InfiniteRange;
   using PolyIsing = model::classical::PolynomialIsing<InfiniteRange>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   PolyIsing poly_ising{InfiniteRange{7}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::METROPOLIS,
                                     solver::RandomNumberEngine::MT,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::MT,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::XORSHIFT,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::MT_64,
                                     solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunSingleFlip(poly_ising, 50, 10, 2, 0.1, 0,
                                     solver::StateUpdateMethod::HEAT_BATH,
                                     solver::RandomNumberEngine::MT_64,
                                     solver::SpinSelectionMethod::SEQUENTIAL));
}

}  // namespace test
}  // namespace compnal

#endif /* OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
