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
//  test_system_poly_ising_chain.hpp
//  compnal
//
//  Created by kohei on 2023/06/25.
//  
//

#ifndef COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_HPP_
#define COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_HPP_

#include "../../../../../include/solver/classical_monte_carlo/system/poly_ising_chain.hpp"


namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloSystem, PolyIsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using PolyIsing = model::classical::PolynomialIsing<Chain>;
   
   std::vector<std::unordered_map<std::int32_t, double>> interaction_set = {
      {{2, -1.0}},
      {{2, +1.0}, {3, -1.0}},
      {{1, -1.5}, {2, +0.5}, {3, +1.0}, {4, -2.0}}
   };
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Chain chain{5, bc};
      for (const auto &interaction: interaction_set) {
         
         PolyIsing poly_ising{chain, interaction};
         const std::int32_t seed = 0;
         
         solver::classical_monte_carlo::System<PolyIsing, std::mt19937> system{poly_ising, seed};
         system.SetSampleByState((std::vector<std::int32_t>{0, 1, 0, 1, 0}));
         
         EXPECT_DOUBLE_EQ(system.ExtractSample()(0), -0.5);
         EXPECT_DOUBLE_EQ(system.ExtractSample()(1), +0.5);
         EXPECT_DOUBLE_EQ(system.ExtractSample()(2), -0.5);
         EXPECT_DOUBLE_EQ(system.ExtractSample()(3), +0.5);
         EXPECT_DOUBLE_EQ(system.ExtractSample()(4), -0.5);
         EXPECT_EQ(system.GenerateCandidateState(0), 1);
         EXPECT_EQ(system.GenerateCandidateState(1), 0);
         EXPECT_EQ(system.GenerateCandidateState(2), 1);
         EXPECT_EQ(system.GenerateCandidateState(3), 0);
         EXPECT_EQ(system.GenerateCandidateState(4), 1);
         EXPECT_EQ(system.GetSystemSize(), 5);
         
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, +0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         
         system.Flip(2, 1);
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, -0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, +0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         system.Flip(0, 1);
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, -0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, -0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, -0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
         EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                          poly_ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, +0.5, +0.5, +0.5}) -
                          poly_ising.CalculateEnergy(system.ExtractSample()));
      }
      
   }
   
   
}


}  // namespace test
}  // namespace compnal


#endif /* COMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_POLY_ISING_HPP_ */