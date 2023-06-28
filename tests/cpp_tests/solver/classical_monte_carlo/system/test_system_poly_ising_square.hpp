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
//  test_system_poly_ising_square.hpp
//  compnal
//
//  Created by kohei on 2023/06/28.
//  
//

#ifndef COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_SQUARE_HPP_
#define COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_SQUARE_HPP_

#include "../../../../../include/solver/classical_monte_carlo/system/poly_ising_square.hpp"


namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloSystem, PolyIsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using PolyIsing = model::classical::PolynomialIsing<Square>;
   
   const std::int32_t x_size = 8;
   const std::int32_t y_size = 8;
   std::vector<std::unordered_map<std::int32_t, double>> interaction_set = {
      {{1, -1.5}, {2, -1.0}},
      {{1, -1.5}, {2, +1.0}, {3, -1.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}, {5, +3.0}, {6, -0.5}},
   };
   std::vector<std::int32_t> initial_state_level(x_size*y_size);
   for (std::int32_t i = 0; i < x_size; ++i) {
      for (std::int32_t j = 0; j < y_size; ++j) {
         initial_state_level[j*x_size + i] = 0;
      }
   }
   
   std::vector<double> initial_state(x_size*y_size);
   for (std::int32_t i = 0; i < x_size; ++i) {
      for (std::int32_t j = 0; j < y_size; ++j) {
         initial_state[j*x_size + i] = -0.5;
      }
   }
   
   Square square{8, 8, BC::OBC};
   PolyIsing poly_ising{square, interaction_set[3]};
   const std::int32_t seed = 0;
   
   solver::classical_monte_carlo::System<PolyIsing, std::mt19937> system{poly_ising, seed};
   system.SetSampleByState(initial_state_level);
   
   for (std::int32_t i = 0; i < x_size; ++i) {
      for (std::int32_t j = 0; j < y_size; ++j) {
         EXPECT_DOUBLE_EQ(system.ExtractSample()(j*x_size + i), -0.5);
         EXPECT_EQ(system.GenerateCandidateState(j*x_size + i), 1);
      }
   }
   
   EXPECT_EQ(system.GetSystemSize(), x_size*y_size);
   
   for (std::int32_t i = 0; i < x_size; ++i) {
      for (std::int32_t j = 0; j < y_size; ++j) {
         initial_state[j*x_size + i] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(j*x_size + i, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         initial_state[j*x_size + i] = -0.5;
         
         system.Flip(0, 1);
         initial_state[0] = 0.5;
         initial_state[j*x_size + i] = 0.5;
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(j*x_size + i, 1),
                          poly_ising.CalculateEnergy(initial_state) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         initial_state[0] = -0.5;
         initial_state[j*x_size + i] = -0.5;
         system.Flip(0, 0);
      }
   }
   
}




}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_SQUARE_HPP_ */
