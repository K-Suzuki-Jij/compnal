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

TEST(Lattice, BaseThreeDimensional) {
   using Lat = lattice::BaseThreeDimensionalLattice;
   using BC = lattice::BoundaryCondition;
   
   EXPECT_THROW((Lat{-1, +9, +2}), std::runtime_error);
   EXPECT_THROW((Lat{+1, -9, +2}), std::runtime_error);
   EXPECT_THROW((Lat{+1, +9, -2}), std::runtime_error);
   EXPECT_THROW((Lat{+2, +2, +2, BC::NONE}), std::runtime_error);
   EXPECT_THROW((Lat{-2, -2, -2, BC::NONE}), std::runtime_error);
   
   EXPECT_EQ((Lat{8, 7, 3}.GetSystemSize()), 8*7*3);
   EXPECT_EQ((Lat{8, 7, 3}.GetBoundaryCondition()), BC::OBC);
   
   EXPECT_EQ((Lat{8, 7, 3, BC::PBC}.GetSystemSize()), 8*7*3);
   EXPECT_EQ((Lat{8, 7, 3, BC::PBC}.GetXSize()), 8);
   EXPECT_EQ((Lat{8, 7, 3, BC::PBC}.GetYSize()), 7);
   EXPECT_EQ((Lat{8, 7, 3, BC::PBC}.GetZSize()), 3);
   EXPECT_EQ((Lat{8, 7, 3, BC::PBC}.GetBoundaryCondition()), BC::PBC);
   
   EXPECT_TRUE((Lat{2, 3, 4}.ValidateCOOIndex({1, 2, 3})));
   EXPECT_TRUE((Lat{2, 3, 4}.ValidateCOOIndex({0, 0, 0})));
   EXPECT_FALSE((Lat{2, 3, 4}.ValidateCOOIndex({1, 3, 3})));
   EXPECT_FALSE((Lat{2, 3, 4}.ValidateCOOIndex({1, 2, 4})));
   EXPECT_FALSE((Lat{2, 3, 4}.ValidateCOOIndex({1, -1, 1})));
   
   EXPECT_EQ((Lat{1, 2, 3}.CalculateIntegerSiteIndex({1, 1, 1})), 4);
   EXPECT_EQ((Lat{4, 3, 3}.CalculateIntegerSiteIndex({2, 1, 2})), 30);
}

TEST(Lattice, Cubic) {
   using Lat = lattice::Cubic;
   std::vector<typename Lat::IndexType> index_list{
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
