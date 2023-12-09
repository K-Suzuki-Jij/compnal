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
//  Created by kohei on 2023/04/30.
//  
//

#pragma once

#include "../../../include/lattice/infinite_range.hpp"

namespace compnal {
namespace test {

TEST(Lattice, InfiniteRange) {
   lattice::InfiniteRange infinite_range{4};
   EXPECT_EQ(infinite_range.GetSystemSize(), 4);
   EXPECT_EQ(infinite_range.GetBoundaryCondition(), lattice::BoundaryCondition::NONE);
   EXPECT_EQ(infinite_range.GenerateCoordinateList(),
             (std::vector<std::tuple<std::int32_t>>{{0}, {1}, {2}, {3}}));
   EXPECT_EQ(infinite_range.CoordinateToInteger(0), 0);
   EXPECT_EQ(infinite_range.CoordinateToInteger(1), 1);
   EXPECT_EQ(infinite_range.CoordinateToInteger(2), 2);
   EXPECT_EQ(infinite_range.CoordinateToInteger(3), 3);
   
   EXPECT_TRUE(infinite_range.ValidateCoordinate(0));
   EXPECT_TRUE(infinite_range.ValidateCoordinate(1));
   EXPECT_TRUE(infinite_range.ValidateCoordinate(2));
   EXPECT_TRUE(infinite_range.ValidateCoordinate(3));
   EXPECT_FALSE(infinite_range.ValidateCoordinate(-1));
   EXPECT_FALSE(infinite_range.ValidateCoordinate(4));
   
   EXPECT_THROW(lattice::InfiniteRange{0}, std::invalid_argument);
   
}

}  // namespace test
}  // namespace compnal

