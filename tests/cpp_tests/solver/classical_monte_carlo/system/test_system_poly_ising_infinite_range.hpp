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
//  test_system_poly_ising_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2023/06/30.
//  
//

#pragma once

#include "../../../../../include/solver/classical_monte_carlo/system/poly_ising_infinite_range.hpp"

namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloSystem, PolyIsingOnInfiniteRange) {
   using InfiniteRange = lattice::InfiniteRange;
   using PolyIsing = model::classical::PolynomialIsing<InfiniteRange>;
   
   const std::int32_t system_size = 10;
   std::vector<std::unordered_map<std::int32_t, double>> interaction_set = {
      {{1, -1.5}, {2, +1.0}},
      {{1, -1.5}, {2, +0.5}, {3, -1.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}, {6, -10}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}, {6, -10}, {7, -3.5}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}, {6, -10}, {7, -3.5}, {8, -1.5}},
   };
   
   Eigen::Vector<double, Eigen::Dynamic> initial_state(system_size);
   for (std::int32_t i = 0; i < system_size; ++i) {
      initial_state[i] = -0.5;
   }
   
   for (const auto &interaction: interaction_set) {
      InfiniteRange infinite_range{system_size};
      PolyIsing poly_ising{infinite_range, interaction};
      
      const std::int32_t seed = 0;
      
      solver::classical_monte_carlo::System<PolyIsing, std::mt19937> system{poly_ising, seed};
      system.SetSampleByValue(initial_state);
      
      for (std::int32_t i = 0; i < system_size; ++i) {
         EXPECT_DOUBLE_EQ(system.ExtractSample()(i), -0.5);
         EXPECT_EQ(system.GenerateCandidateState(i), 1);
      }
      
      EXPECT_EQ(system.GetSystemSize(), system_size);
      
      for (std::int32_t i = 1; i < system_size; ++i) {
         initial_state[i] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergy(), poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(i, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(0, 1);
         EXPECT_DOUBLE_EQ(system.GetEnergy(), poly_ising.CalculateEnergy(system.ExtractSample()));
         initial_state[0] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(i, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(4, 1);
         EXPECT_DOUBLE_EQ(system.GetEnergy(), poly_ising.CalculateEnergy(system.ExtractSample()));
         initial_state[4] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(i, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(0, 0);
         system.Flip(4, 0);
         initial_state[i] = -0.5;
         initial_state[0] = -0.5;
         initial_state[4] = -0.5;
      }
   }
}


}  // namespace test
}  // namespace compnal
