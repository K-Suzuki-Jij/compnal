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
//  test_two_dimensional_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_TEST_LATTICE_TWO_DIMENSIONAL_LATTICE_HPP_
#define COMPNAL_TEST_LATTICE_TWO_DIMENSIONAL_LATTICE_HPP_

#include "../../src/lattice/base_two_dimensional_lattice.hpp"
#include "../../src/lattice/square.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(Lattice, BaseTwoDimensionalLattice) {
   using Lat = lattice::BaseTwoDimensionalLattice;
   using BC = lattice::BoundaryCondition;
   
   EXPECT_THROW((Lat{-1, +9}), std::runtime_error);
   EXPECT_THROW((Lat{+3, -1}), std::runtime_error);
   EXPECT_THROW((Lat{+2, +1, BC::NONE}), std::runtime_error);
   
   EXPECT_EQ((Lat{8, 7}.GetSystemSize()), 56);
   EXPECT_EQ((Lat{8, 7}.GetXSize()), 8);
   EXPECT_EQ((Lat{8, 7}.GetYSize()), 7);
   EXPECT_EQ((Lat{8, 7}.GetBoundaryCondition()), BC::OBC);
   EXPECT_EQ((Lat{8, 7, BC::PBC}.GetBoundaryCondition()), BC::PBC);

   EXPECT_TRUE((Lat{3, 4}.ValidateCOOIndex({2, 3})));
   EXPECT_TRUE((Lat{3, 4}.ValidateCOOIndex({0, 0})));
   EXPECT_FALSE((Lat{3, 4}.ValidateCOOIndex({2, 4})));
   EXPECT_FALSE((Lat{3, 4}.ValidateCOOIndex({-1, 0})));
   
   EXPECT_EQ((Lat{2, 4}.CalculateIntegerSiteIndex({1, 1})), 3);
   EXPECT_EQ((Lat{2, 3}.CalculateIntegerSiteIndex({1, 1})), 3);
   EXPECT_EQ((Lat{3, 3}.CalculateIntegerSiteIndex({1, 1})), 4);
}

TEST(Lattice, Square) {
   using Lat = lattice::Square;
   std::vector<typename Lat::COOIndexType> index_list{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}};
   EXPECT_EQ((Lat{2, 3}.GenerateIndexList()), index_list);
}



} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_LATTICE_TWO_DIMENSIONAL_LATTICE_HPP_ */
