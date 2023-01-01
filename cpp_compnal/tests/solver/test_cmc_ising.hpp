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
//  test_cmc_ising.hpp
//  compnal
//
//  Created by kohei on 2022/09/25.
//  
//

#ifndef COMPNAL_TEST_SOLVER_CMC_ISING_HPP_
#define COMPNAL_TEST_SOLVER_CMC_ISING_HPP_


#include "../../src/solver/classical_monte_carlo.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {


TEST(SolverClassicalMonteCarlo, IsingChainOBC) {
   const std::int32_t system_size = 10;
   const lattice::Chain lattice(system_size, lattice::BoundaryCondition::OBC);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(5);
   solver.SetTemperature(0.01);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)), system_size);
   }
}

TEST(SolverClassicalMonteCarlo, IsingChainPBC) {
   const std::int32_t system_size = 10;
   const lattice::Chain lattice(system_size, lattice::BoundaryCondition::PBC);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(5);
   solver.SetTemperature(0.01);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)), system_size);
   }
}

TEST(SolverClassicalMonteCarlo, IsingSquareOBC) {
   const std::int32_t x_size = 3;
   const std::int32_t y_size = 4;
   const lattice::Square lattice(x_size, y_size, lattice::BoundaryCondition::OBC);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(1000);
   solver.SetNumSamples(8);
   solver.SetTemperature(0.5);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)), x_size*y_size);
   }
   
}

TEST(SolverClassicalMonteCarlo, IsingSquarePBC) {
   const std::int32_t x_size = 3;
   const std::int32_t y_size = 5;
   const lattice::Square lattice(x_size, y_size, lattice::BoundaryCondition::PBC);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(8);
   solver.SetTemperature(0.5);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)), x_size*y_size);
   }
}

TEST(SolverClassicalMonteCarlo, IsingCubicOBC) {
   const std::int32_t x_size = 3;
   const std::int32_t y_size = 4;
   const std::int32_t z_size = 5;
   const lattice::Cubic lattice(x_size, y_size, z_size, lattice::BoundaryCondition::OBC);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(8);
   solver.SetTemperature(0.1);
   solver.Run(seed);
   
   std::int32_t gs_count = 0;
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      if (std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)) == model.GetSystemSize()) {
         gs_count++;
      }
   }
   
   EXPECT_GE(gs_count, 4);
}

TEST(SolverClassicalMonteCarlo, IsingCubicPBC) {
   const std::int32_t x_size = 3;
   const std::int32_t y_size = 4;
   const std::int32_t z_size = 5;
   const lattice::Cubic lattice(x_size, y_size, z_size, lattice::BoundaryCondition::PBC);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(8);
   solver.SetTemperature(0.1);
   solver.Run(seed);
   
   std::int32_t gs_count = 0;
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      if (std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)) == model.GetSystemSize()) {
         gs_count++;
      }
   }
   
   EXPECT_GE(gs_count, 4);
}


TEST(SolverClassicalMonteCarlo, IsingInfiniteRange) {
   const std::int32_t system_size = 10;
   const lattice::InfiniteRange lattice(system_size);
   model::Ising model(lattice, 0.0, -1.0);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(1000);
   solver.SetNumSamples(8);
   solver.SetTemperature(0.1);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)), system_size);
   }
   
}

TEST(SolverClassicalMonteCarlo, IsingAnyLattice) {
   
   using LinearType = model::Ising<lattice::AnyLattice, double>::LinearType;
   using QuadraticType = model::Ising<lattice::AnyLattice, double>::QuadraticType;
   const std::int32_t system_size = 10;
   
   QuadraticType quad;
   
   for (std::int32_t i = 0; i < system_size; ++i) {
      for (std::int32_t j = i + 1; j < system_size; ++j) {
         quad[{i, j}] = -0.1;
      }
   }
   
   const std::uint64_t seed = 1;
   auto model = model::make_ising<lattice::AnyLattice, double>(lattice::AnyLattice{}, LinearType{}, quad);
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(10);
   solver.SetTemperature(0.15);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::abs(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)), system_size);
   }

}



} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_SOLVER_CMC_ISING_HPP_ */
