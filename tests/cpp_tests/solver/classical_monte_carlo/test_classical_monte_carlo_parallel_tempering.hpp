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
//  test_classicalmonte_carlo_parallel_tempering.hpp
//  compnal
//
//  Created by kohei on 2023/07/08.
//  
//

#pragma once

#include "../../../../include/lattice/all.hpp"
#include "../../../../include/model/classical/ising.hpp"
#include "../../../../include/solver/classical_monte_carlo/classical_monte_carlo.hpp"

namespace compnal {
namespace test {


TEST(SolverClassicalMonteCarloPT, IsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using Ising = model::classical::Ising<Chain>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      const Ising ising{Chain{3, bc}, -1.0, -4.0, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::METROPOLIS,
                                               solver::RandomNumberEngine::MT,
                                               solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::MT_64,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::XORSHIFT,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
   }
   
}

TEST(SolverClassicalMonteCarloPT, IsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::Ising<Square>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Square{3, 4, bc}, -1.0, -4.0, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::METROPOLIS,
                                               solver::RandomNumberEngine::MT,
                                               solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::MT_64,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::XORSHIFT,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
   }
   
}

TEST(SolverClassicalMonteCarloPT, IsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using Ising = model::classical::Ising<Cubic>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Cubic{3, 4, 2, bc}, -1.0, -4.0, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::METROPOLIS,
                                               solver::RandomNumberEngine::MT,
                                               solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::MT_64,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::XORSHIFT,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
   }
   
}

TEST(SolverClassicalMonteCarloPT, IsingOnInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using InfiniteRange = lattice::InfiniteRange;
   using Ising = model::classical::Ising<InfiniteRange>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<Ising>();
   
   Ising ising{InfiniteRange{3}, -1.0, -4.0, 1.5, 2};
   
   EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                            Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                            0,
                                            solver::StateUpdateMethod::METROPOLIS,
                                            solver::RandomNumberEngine::MT,
                                            solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                            Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                            0,
                                            solver::StateUpdateMethod::HEAT_BATH,
                                            solver::RandomNumberEngine::MT_64,
                                            solver::SpinSelectionMethod::SEQUENTIAL));
   
   EXPECT_NO_THROW(cmc.RunParallelTempering(ising, 10, 10, 2, 2,
                                            Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                            0,
                                            solver::StateUpdateMethod::HEAT_BATH,
                                            solver::RandomNumberEngine::XORSHIFT,
                                            solver::SpinSelectionMethod::SEQUENTIAL));
   
}

TEST(SolverClassicalMonteCarloPT, PolyIsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using PolyIsing = model::classical::PolynomialIsing<Chain>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Chain{7, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::METROPOLIS,
                                               solver::RandomNumberEngine::MT,
                                               solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::MT_64,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::XORSHIFT,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
   }
   
}

TEST(SolverClassicalMonteCarloPT, PolyIsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using PolyIsing = model::classical::PolynomialIsing<Square>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Square{7, 6, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::METROPOLIS,
                                               solver::RandomNumberEngine::MT,
                                               solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::MT_64,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::XORSHIFT,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
   }
   
}

TEST(SolverClassicalMonteCarloPT, PolyIsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using PolyIsing = model::classical::PolynomialIsing<Cubic>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Cubic{7, 6, 5, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::METROPOLIS,
                                               solver::RandomNumberEngine::MT,
                                               solver::SpinSelectionMethod::RANDOM));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::MT_64,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
      
      EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                               Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                               0,
                                               solver::StateUpdateMethod::HEAT_BATH,
                                               solver::RandomNumberEngine::XORSHIFT,
                                               solver::SpinSelectionMethod::SEQUENTIAL));
   }
   
}

TEST(SolverClassicalMonteCarloPT, PolyIsingOnInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using InfiniteRange = lattice::InfiniteRange;
   using PolyIsing = model::classical::PolynomialIsing<InfiniteRange>;
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo<PolyIsing>();
   
   PolyIsing poly_ising{InfiniteRange{7}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
   
   EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                            Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                            0,
                                            solver::StateUpdateMethod::METROPOLIS,
                                            solver::RandomNumberEngine::MT,
                                            solver::SpinSelectionMethod::RANDOM));
   
   EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                            Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                            0,
                                            solver::StateUpdateMethod::HEAT_BATH,
                                            solver::RandomNumberEngine::MT_64,
                                            solver::SpinSelectionMethod::SEQUENTIAL));
   
   EXPECT_NO_THROW(cmc.RunParallelTempering(poly_ising, 10, 10, 2, 2,
                                            Eigen::Vector<double, 3>{1.0, 2.0, 3.0},
                                            0,
                                            solver::StateUpdateMethod::HEAT_BATH,
                                            solver::RandomNumberEngine::XORSHIFT,
                                            solver::SpinSelectionMethod::SEQUENTIAL));
   
}

}  // namespace test
}  // namespace compnal
