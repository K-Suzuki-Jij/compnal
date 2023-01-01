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
//  test_base_three_dimensional_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_TEST_LATTICE_THREE_DIMENSIONAL_LATTICE_HPP_
#define COMPNAL_TEST_LATTICE_THREE_DIMENSIONAL_LATTICE_HPP_

#include "../../src/lattice/base_three_dimensional_lattice.hpp"
#include "../../src/lattice/cubic.hpp"

namespace compnal {
namespace test {

TEST(LatticeBaseThreeDimensionalLattice, Constructor) {
   
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{8, 7, 3}.GetSystemSize()), 8*7*3);
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{8, 7, 3}.GetBoundaryCondition()), lattice::BoundaryCondition::OBC);
   
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{
      8, 7, 3, lattice::BoundaryCondition::PBC
   }.GetSystemSize()), 8*7*3);
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{
      8, 7, 3, lattice::BoundaryCondition::PBC
   }.GetXSize()), 8);
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{
      8, 7, 3, lattice::BoundaryCondition::PBC
   }.GetYSize()), 7);
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{
      8, 7, 3, lattice::BoundaryCondition::PBC
   }.GetZSize()), 3);
   EXPECT_EQ((lattice::BaseThreeDimensionalLattice{
      8, 7, 3, lattice::BoundaryCondition::PBC
   }.GetBoundaryCondition()), lattice::BoundaryCondition::PBC);
   
   EXPECT_THROW((lattice::BaseThreeDimensionalLattice{-1, +9, +2}), std::runtime_error);
   EXPECT_THROW((lattice::BaseThreeDimensionalLattice{+1, -9, +2}), std::runtime_error);
   EXPECT_THROW((lattice::BaseThreeDimensionalLattice{+1, +9, -2}), std::runtime_error);
   
   EXPECT_THROW((lattice::BaseThreeDimensionalLattice{+2, +2, +2, lattice::BoundaryCondition::NONE}), std::runtime_error);
   EXPECT_THROW((lattice::BaseThreeDimensionalLattice{-2, -2, -2, lattice::BoundaryCondition::NONE}), std::runtime_error);
   
}

TEST(LatticeCubic, Basic) {
   auto index_list = std::vector<std::tuple<std::int32_t, std::int32_t, std::int32_t>>{
      {0, 0, 0}, {0, 0, 1},
      {0, 1, 0}, {0, 1, 1},
      {0, 2, 0}, {0, 2, 1},
      {1, 0, 0}, {1, 0, 1},
      {1, 1, 0}, {1, 1, 1},
      {1, 2, 0}, {1, 2, 1}
   };
   EXPECT_EQ((lattice::Cubic{2, 3, 2}.GenerateIndexList()), index_list);
}

}
}

#endif /* COMPNAL_TEST_LATTICE_THREE_DIMENSIONAL_LATTICE_HPP_ */
