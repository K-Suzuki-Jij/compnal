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
//  test_system_ising_square.hpp
//  compnal
//
//  Created by kohei on 2023/06/13.
//  
//

#pragma once

#include "../../../../../include/solver/classical_monte_carlo/system/ising_square.hpp"

namespace compnal {
namespace test {

TEST(SolverClassicalMonteCarloSystem, IsingOnSquareSpinHalf) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::Ising<Square>;
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Square square{2, 3, bc};
      Ising ising{square, 1.0, -4.0};
      const std::int32_t seed = 0;
      
      solver::classical_monte_carlo::System<Ising, std::mt19937> system{ising, seed};
      system.SetSampleByValue(Eigen::Vector<double, 6>(-0.5, +0.5, -0.5, -0.5, -0.5, +0.5));
      EXPECT_DOUBLE_EQ(system.ExtractSample()(0), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(1), +0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(2), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(3), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(4), -0.5);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(5), +0.5);
      EXPECT_EQ(system.GenerateCandidateState(0), 1);
      EXPECT_EQ(system.GenerateCandidateState(1), 0);
      EXPECT_EQ(system.GenerateCandidateState(2), 1);
      EXPECT_EQ(system.GenerateCandidateState(3), 1);
      EXPECT_EQ(system.GenerateCandidateState(4), 1);
      EXPECT_EQ(system.GenerateCandidateState(5), 0);
      
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, -0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, +0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      system.Flip(3, 1);
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, -0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, +0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      system.Flip(3, 0);
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, -0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, -0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{+0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, +0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, +0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, +0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-0.5, +0.5, -0.5, -0.5, -0.5, +0.5}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
   }
}

TEST(SolverClassicalMonteCarloSystem, IsingOnSquareSpinOne) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::Ising<Square>;
   
   const auto check_include = [](const std::int32_t val, const std::vector<std::int32_t> &list) {
      for (const auto &it: list) {
         if (val == it) {
            return true;
         }
      }
      return false;
   };
   
   for (const auto &bc: std::vector<BC>{BC::OBC, BC::PBC}) {
      Square square{2, 3, bc};
      Ising ising{square, 1.0, -4.0, 1};
      const std::int32_t seed = 0;
      
      solver::classical_monte_carlo::System<Ising, std::mt19937> system{ising, seed};
      system.SetSampleByValue(Eigen::Vector<double, 6>(-1, 1, 1, 0, 0, -1));
      EXPECT_DOUBLE_EQ(system.ExtractSample()(0), -1);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(1), +1);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(2), +1);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(3), +0);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(4), +0);
      EXPECT_DOUBLE_EQ(system.ExtractSample()(5), -1);
      EXPECT_TRUE(check_include(system.GenerateCandidateState(0), {1, 2}));
      EXPECT_TRUE(check_include(system.GenerateCandidateState(1), {0, 1}));
      EXPECT_TRUE(check_include(system.GenerateCandidateState(2), {0, 1}));
      EXPECT_TRUE(check_include(system.GenerateCandidateState(3), {0, 2}));
      EXPECT_TRUE(check_include(system.GenerateCandidateState(4), {0, 2}));
      EXPECT_TRUE(check_include(system.GenerateCandidateState(5), {1, 2}));
      
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, -1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, -1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, -1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, -1, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{0, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 0, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 0, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, 0}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      system.Flip(3, 2);
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, -1, 1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, -1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, -1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, +1, -1, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{0, 1, 1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 0, 1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 0, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, +1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, +1, 0, 0}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      system.Flip(3, 1);
      EXPECT_DOUBLE_EQ(system.GetEnergy(), ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, -1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, -1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, -1, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, -1, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 0),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(0, 1),
                       ising.CalculateEnergy(std::vector<double>{0, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(1, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 0, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(2, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 0, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(3, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(4, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, -1}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      EXPECT_DOUBLE_EQ(system.GetEnergyDifference(5, 1),
                       ising.CalculateEnergy(std::vector<double>{-1, 1, 1, 0, 0, 0}) -
                       ising.CalculateEnergy(system.ExtractSample()));
      
   }
}

}  // namespace test
}  // namespace compnal
