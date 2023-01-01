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
//  Created by Kohei Suzuki on 2022/06/11.
//

#ifndef COMPNAL_TEST_SOLVER_CMC_POLYNOMIAL_ISING_HPP_
#define COMPNAL_TEST_SOLVER_CMC_POLYNOMIAL_ISING_HPP_

#include "../../src/solver/classical_monte_carlo.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarlo, PolyIsingAnyLattice) {
   using PolynomialType = model::PolynomialIsing<lattice::AnyLattice, double>::PolynomialType;
   PolynomialType interaction;
   const std::int32_t system_size = 10;
   for (std::int32_t i = 0; i < system_size; ++i) {
      for (std::int32_t j = i + 1; j < system_size; ++j) {
         for (std::int32_t k = j + 1; k < system_size; ++k) {
            interaction[{i, j, k}] = -0.03;
         }
      }
   }
   
   const std::uint64_t seed = 1;
   model::PolynomialIsing<lattice::AnyLattice, double> model{lattice::AnyLattice{}, interaction};
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(10);
   solver.SetTemperature(0.15);
   solver.Run(seed);

   EXPECT_EQ(solver.CalculateAverage(), 1);
}

TEST(SolverClassicalMonteCarlo, PolyIsingChain) {
   const lattice::Chain lattice(9, lattice::BoundaryCondition::PBC);
   const std::unordered_map<std::int32_t, double> interaction{{2, -1.0}, {1, -1.0}};
   model::PolynomialIsing model(lattice, interaction);
   
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(1000);
   solver.SetNumSamples(5);
   solver.SetTemperature(0.1);
   solver.Run(seed);
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      EXPECT_EQ(std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0.0), 9.0);
   }
}

TEST(SolverClassicalMonteCarlo, PolyIsingCubicPBC) {
   const lattice::Cubic lattice(3, 3, 3, lattice::BoundaryCondition::PBC);
   const std::unordered_map<std::int32_t, double> interaction{{3, -1}};
   model::PolynomialIsing model(lattice, interaction);
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(10);
   solver.SetTemperature(0.4);
   solver.Run();
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      for (const auto &it: solver.GetSamples().at(i)) {
         printf("%+d, ", it);
      }
      printf("SUM=%lf", std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)/27.0);
      printf("\n");
   }
   
   printf("%lf\n", solver.CalculateAverage());
}

TEST(SolverClassicalMonteCarlo, PolyIsingCubicOBC) {
   const lattice::Cubic lattice(3, 3, 3, lattice::BoundaryCondition::OBC);
   const std::unordered_map<std::int32_t, double> interaction{{3, -1}};
   model::PolynomialIsing model(lattice, interaction);
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(10);
   solver.SetTemperature(0.4);
   solver.Run();
   
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      for (const auto &it: solver.GetSamples().at(i)) {
         printf("%+d, ", it);
      }
      printf("SUM=%lf", std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0)/27.0);
      printf("\n");
   }
   
   printf("%lf\n", solver.CalculateAverage());
}


TEST(SolverClassicalMonteCarlo, PolyIsingSquarePBC) {
   const lattice::Square lattice(6, 6, lattice::BoundaryCondition::PBC);
   const std::unordered_map<std::int32_t, double> interaction{{2, -1}};
   model::PolynomialIsing model(lattice, interaction);
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(1);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(1);
   solver.SetTemperature(0.01);
   solver.Run(seed);
   
   std::int32_t gs_count = 0;
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      const auto mag = std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0);
      if (mag == solver.GetSamples().at(i).size() || -1*mag == solver.GetSamples().at(i).size()) {
         gs_count++;
      }
   }
   EXPECT_GE(gs_count, 1);
}

TEST(SolverClassicalMonteCarlo, PolyIsingSquareOBC) {
   const lattice::Square lattice(6, 6, lattice::BoundaryCondition::OBC);
   const std::unordered_map<std::int32_t, double> interaction{{2, -1}};
   model::PolynomialIsing model(lattice, interaction);
   const std::uint64_t seed = 1;
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(1);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(1);
   solver.SetTemperature(0.01);
   solver.Run(seed);
   
   std::int32_t gs_count = 0;
   for (std::size_t i = 0; i < solver.GetSamples().size(); ++i) {
      const auto mag = std::accumulate(solver.GetSamples().at(i).begin(), solver.GetSamples().at(i).end(), 0);
      if (mag == solver.GetSamples().at(i).size() || -1*mag == solver.GetSamples().at(i).size()) {
         gs_count++;
      }
   }
   EXPECT_GE(gs_count, 1);
}

TEST(SolverClassicalMonteCarlo, PolyIsingInfinitRange) {
   const std::unordered_map<std::int32_t, double> interaction{{3, -0.03}};
   const std::uint64_t seed = 1;
   model::PolynomialIsing model{lattice::InfiniteRange{10}, interaction};
   solver::ClassicalMonteCarlo solver(model);
   solver.SetNumThreads(4);
   solver.SetNumSweeps(10000);
   solver.SetNumSamples(10);
   solver.SetTemperature(0.15);
   solver.Run(seed);
   
   EXPECT_EQ(solver.CalculateAverage(), 1);
}


} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_SOLVER_CMC_POLYNOMIAL_ISING_HPP_ */
