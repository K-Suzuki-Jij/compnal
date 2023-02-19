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
//  test_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_TEST_LATTICE_INFINITE_RANGE_HPP_
#define COMPNAL_TEST_LATTICE_INFINITE_RANGE_HPP_

#include "../../src/lattice/infinite_range.hpp"
#include "../../src/lattice/cubic.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(Lattice, InfiniteRange) {
   using Lat = lattice::InfiniteRange;
   using BC = lattice::BoundaryCondition;
   
   EXPECT_THROW(Lat{-1}, std::runtime_error);
   EXPECT_EQ(Lat{2}.GetSystemSize(), 2);
   EXPECT_EQ(Lat{2}.GetBoundaryCondition(), BC::NONE);
   
   EXPECT_TRUE(Lat{2}.ValidateCOOIndex(0));
   EXPECT_TRUE(Lat{2}.ValidateCOOIndex(1));
   EXPECT_FALSE(Lat{2}.ValidateCOOIndex(2));
   EXPECT_FALSE(Lat{2}.ValidateCOOIndex(-1));
   
   EXPECT_EQ(Lat{3}.CalculateIntegerSiteIndex(0), 0);
   EXPECT_EQ(Lat{3}.CalculateIntegerSiteIndex(2), 2);
   
   std::vector<Lat::IndexType> index_list{0, 1, 2, 3};
   EXPECT_EQ(Lat{4}.GenerateIndexList(), index_list);
}


} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_LATTICE_INFINITE_RANGE_HPP_ */
