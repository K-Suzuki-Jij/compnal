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
   
   auto cmc = solver::classical_monte_carlo::make_classical_monte_carlo(ising);
   cmc.SetNumSweeps(500);
   cmc.SetNumSamples(100);
   cmc.SetNumThreads(2);
   cmc.SetTemperature(10);
   cmc.SetStateUpdateMethod(solver::StateUpdateMethod::HEAT_BATH);
   cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT);
   cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::SEQUENTIAL);
   
   EXPECT_EQ(cmc.GetNumSweeps(), 500);
   EXPECT_EQ(cmc.GetNumSamples(), 100);
   EXPECT_EQ(cmc.GetNumThreads(), 2);
   EXPECT_DOUBLE_EQ(cmc.GetTemperature(), 10.0);
   EXPECT_EQ(cmc.GetStateUpdateMethod(), solver::StateUpdateMethod::HEAT_BATH);
   EXPECT_EQ(cmc.GetRandomNumberEngine(), solver::RandomNumberEngine::MT);
   EXPECT_EQ(cmc.GetSpinSelectionMethod(), solver::SpinSelectionMethod::SEQUENTIAL);
   
   cmc.RunSampling();
   EXPECT_EQ(cmc.GetSamples().rows(), cmc.GetNumSamples());
   EXPECT_EQ(cmc.CalculateEnergies().size(), cmc.GetNumSamples());
   
   cmc.RunSampling(0);
   EXPECT_EQ(cmc.GetSeed(), 0);
   
   EXPECT_THROW(cmc.SetNumSweeps(-1), std::invalid_argument);
   EXPECT_THROW(cmc.SetNumSamples(0), std::invalid_argument);
   EXPECT_THROW(cmc.SetNumThreads(0), std::invalid_argument);
   EXPECT_THROW(cmc.SetTemperature(-0.3), std::invalid_argument);
   
}

TEST(SolverClassicalMonteCarlo, IsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using Ising = model::classical::Ising<Chain>;
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Chain{3, bc}, -1.0, -4.0, 1.5, 2};
      
      auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo(ising);
      cmc.SetNumSweeps(500);
      cmc.SetNumSamples(10);
      cmc.SetNumThreads(2);
      cmc.SetTemperature(0.1);
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::METROPOLIS);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::HEAT_BATH);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::XORSHIFT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT_64);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::RANDOM);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::SEQUENTIAL);
      EXPECT_NO_THROW(cmc.RunSampling());
   }
}

TEST(SolverClassicalMonteCarlo, IsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::Ising<Square>;
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Square{3, 4, bc}, -1.0, -4.0, 1.5, 2};
      
      auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo(ising);
      cmc.SetNumSweeps(500);
      cmc.SetNumSamples(10);
      cmc.SetNumThreads(2);
      cmc.SetTemperature(0.1);
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::METROPOLIS);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::HEAT_BATH);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::XORSHIFT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT_64);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::RANDOM);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::SEQUENTIAL);
      EXPECT_NO_THROW(cmc.RunSampling());
   }
}

TEST(SolverClassicalMonteCarlo, IsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using Ising = model::classical::Ising<Cubic>;
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Ising ising{Cubic{3, 4, 2, bc}, -1.0, -4.0, 1.5, 2};
      
      auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo(ising);
      cmc.SetNumSweeps(500);
      cmc.SetNumSamples(10);
      cmc.SetNumThreads(2);
      cmc.SetTemperature(0.1);
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::METROPOLIS);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::HEAT_BATH);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::XORSHIFT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT_64);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::RANDOM);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::SEQUENTIAL);
      EXPECT_NO_THROW(cmc.RunSampling());
   }
}

TEST(SolverClassicalMonteCarlo, IsingOnInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using InfiniteRange = lattice::InfiniteRange;
   using Ising = model::classical::Ising<InfiniteRange>;
   
   Ising ising{InfiniteRange{3}, -1.0, -4.0, 1.5, 2};
   
   auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo(ising);
   cmc.SetNumSweeps(500);
   cmc.SetNumSamples(10);
   cmc.SetNumThreads(2);
   cmc.SetTemperature(0.1);
   
   cmc.SetStateUpdateMethod(solver::StateUpdateMethod::METROPOLIS);
   EXPECT_NO_THROW(cmc.RunSampling());
   
   cmc.SetStateUpdateMethod(solver::StateUpdateMethod::HEAT_BATH);
   EXPECT_NO_THROW(cmc.RunSampling());
   
   cmc.SetRandomNumberEngine(solver::RandomNumberEngine::XORSHIFT);
   EXPECT_NO_THROW(cmc.RunSampling());
   
   cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT);
   EXPECT_NO_THROW(cmc.RunSampling());
   
   cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT_64);
   EXPECT_NO_THROW(cmc.RunSampling());
   
   cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::RANDOM);
   EXPECT_NO_THROW(cmc.RunSampling());
   
   cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::SEQUENTIAL);
   EXPECT_NO_THROW(cmc.RunSampling());
}


TEST(SolverClassicalMonteCarlo, PolyIsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using PolyIsing = model::classical::PolynomialIsing<Chain>;
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      PolyIsing poly_ising{Chain{7, bc}, {{1, -1.5}, {4, +2.0}}, 1.5, 2};
      
      auto cmc = solver::classical_monte_carlo::ClassicalMonteCarlo(poly_ising);
      cmc.SetNumSweeps(500);
      cmc.SetNumSamples(10);
      cmc.SetNumThreads(2);
      cmc.SetTemperature(0.1);
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::METROPOLIS);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetStateUpdateMethod(solver::StateUpdateMethod::HEAT_BATH);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::XORSHIFT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetRandomNumberEngine(solver::RandomNumberEngine::MT_64);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::RANDOM);
      EXPECT_NO_THROW(cmc.RunSampling());
      
      cmc.SetSpinSelectionMethod(solver::SpinSelectionMethod::SEQUENTIAL);
      EXPECT_NO_THROW(cmc.RunSampling());
   }
}

}  // namespace test
}  // namespace compnal

#endif /* OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_HPP_ */
