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
//  test_system_poly_ising_cubic.hpp
//  compnal
//
//  Created by kohei on 2023/07/02.
//  
//

#ifndef COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_CUBIC_HPP_
#define COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_CUBIC_HPP_

#include "../../../../../include/solver/classical_monte_carlo/system/poly_ising_cubic.hpp"

namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloSystem, PolyIsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using PolyIsing = model::classical::PolynomialIsing<Cubic>;
   
   const std::int32_t x_size = 6;
   const std::int32_t y_size = 6;
   const std::int32_t z_size = 6;
   const std::int32_t system_size = x_size*y_size*z_size;
   std::vector<std::unordered_map<std::int32_t, double>> interaction_set = {
      {{2, +1.0}},
      {{1, -1.5}, {2, +0.5}, {3, -1.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}, {6, -10}}
   };
   
   std::vector<std::int32_t> initial_state_level(system_size);
   for (std::int32_t i = 0; i < system_size; ++i) {
      initial_state_level[i] = 0;
   }
   
   std::vector<double> initial_state(system_size);
   for (std::int32_t i = 0; i < system_size; ++i) {
      initial_state[i] = -0.5;
   }
   
   std::random_device rand;
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      for (const auto &interaction: interaction_set) {
         Cubic cubic{x_size, y_size, z_size, bc};
         PolyIsing poly_ising{cubic, interaction};
         
         const std::int32_t seed = 0;
         
         solver::classical_monte_carlo::System<PolyIsing, std::mt19937> system{poly_ising, seed};
         system.SetSampleByState(initial_state_level);
         
         for (std::int32_t i = 0; i < x_size; ++i) {
            for (std::int32_t j = 0; j < y_size; ++j) {
               for (std::int32_t k = 0; k < z_size; ++k) {
                  EXPECT_DOUBLE_EQ(system.ExtractSample()(k*y_size*x_size + j*x_size + i), -0.5);
                  EXPECT_EQ(system.GenerateCandidateState(k*y_size*x_size + j*x_size + i), 1);
               }
            }
         }
         
         EXPECT_EQ(system.GetSystemSize(), x_size*y_size*z_size);
         
         const std::int32_t rand_index = rand()%system_size;
         initial_state[rand_index] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(rand_index, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(0, 1);
         initial_state[0] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(rand_index, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         
         system.Flip(system_size - 1, 1);
         initial_state[system_size - 1] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(rand_index, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(system_size/2, 1);
         initial_state[system_size/2] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(rand_index, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(0, 0);
         system.Flip(system_size - 1, 0);
         system.Flip(system_size/2, 0);
         initial_state[rand_index] = -0.5;
         initial_state[0] = -0.5;
         initial_state[system_size - 1] = -0.5;
         initial_state[system_size/2] = -0.5;
      }
   }
   
}

}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_CUBIC_HPP_ */
