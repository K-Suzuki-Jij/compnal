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
//  test_system_ising_cubic.hpp
//  compnal
//
//  Created by kohei on 2023/06/13.
//  
//

#ifndef OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_ISING_CUBIC_HPP_
#define OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_ISING_CUBIC_HPP_


#include "../../../../../include/solver/classical_monte_carlo/system/ising_cubic.hpp"

namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloSystem, IsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using Ising = model::classical::Ising<Cubic>;
   
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Cubic cubic{2, 3, 2, bc};
      Ising ising{cubic, 1.0, -4.0};
      const std::int32_t seed = 0;
      solver::classical_monte_carlo::System<Ising, std::mt19937> system{ising, seed};
      system.SetSampleByState((std::vector<std::int32_t>{0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1}));
      EXPECT_DOUBLE_EQ(system.ExtractSample()(0), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(1), +0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(2), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(3), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(4), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(5), +0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(6), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(7), +0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(8), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(9), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(10), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(11), +0.5);
      
      EXPECT_EQ(system.GenerateCandidateState(0), 1);
      EXPECT_EQ(system.GenerateCandidateState(1), 0);
      EXPECT_EQ(system.GenerateCandidateState(2), 1);
      EXPECT_EQ(system.GenerateCandidateState(3), 1);
      EXPECT_EQ(system.GenerateCandidateState(4), 1);
      EXPECT_EQ(system.GenerateCandidateState(5), 0);
      EXPECT_EQ(system.GenerateCandidateState(6), 1);
      EXPECT_EQ(system.GenerateCandidateState(7), 0);
      EXPECT_EQ(system.GenerateCandidateState(8), 1);
      EXPECT_EQ(system.GenerateCandidateState(9), 1);
      EXPECT_EQ(system.GenerateCandidateState(10), 1);
      EXPECT_EQ(system.GenerateCandidateState(11), 0);
      
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, -0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(6, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(7, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, -0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(8, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(9, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(10, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(11, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, -0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(6, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, +0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(7, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(8, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, +0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(9, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(10, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, +0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(11, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      system.Flip(3, 1);
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(6, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(7, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(8, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(9, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(10, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(11, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, -0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, +0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(6, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, +0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(7, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(8, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, +0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(9, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(10, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, +0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(11, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      system.Flip(3, 0);
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, -0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(6, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(7, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, -0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(8, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(9, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(10, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(11, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, -0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(6, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, +0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(7, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(8, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, +0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(9, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(10, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, +0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(11, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
   }
   
   
}




}  // namespace test
}  // namespace compnal


#endif /* OMPNAL_TEST_SOLVER_CLASSICAL_MONTE_CARLO_SYSTEM_ISING_CUBIC_HPP_ */
