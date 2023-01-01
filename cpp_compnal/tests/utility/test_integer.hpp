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
//  Created by Kohei Suzuki on 2022/07/06.
//

#ifndef COMPNAL_TEST_UTILITY_INTEGER_HPP_
#define COMPNAL_TEST_UTILITY_INTEGER_HPP_

#include "../../src/utility/integer.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(UtilityInteger, GenerateAllCombinations) {
   
   std::vector<std::vector<int>> combinations;
   std::vector<int> seed = {1, 2, 3, 4};
   
   combinations = utility::GenerateAllCombinations(seed, 0);
   EXPECT_EQ(combinations.size(), 0);
   
   combinations = utility::GenerateAllCombinations(seed, 1);
   EXPECT_EQ(combinations.size(), 4);
   EXPECT_EQ(combinations[0].size(), 1);
   EXPECT_EQ(combinations[1].size(), 1);
   EXPECT_EQ(combinations[2].size(), 1);
   EXPECT_EQ(combinations[3].size(), 1);
   EXPECT_EQ(combinations[0][0], 1);
   EXPECT_EQ(combinations[1][0], 2);
   EXPECT_EQ(combinations[2][0], 3);
   EXPECT_EQ(combinations[3][0], 4);
   
   combinations = utility::GenerateAllCombinations(seed, 3);
   EXPECT_EQ(combinations.size()   , 4);
   EXPECT_EQ(combinations[0].size(), 3);
   EXPECT_EQ(combinations[1].size(), 3);
   EXPECT_EQ(combinations[2].size(), 3);
   EXPECT_EQ(combinations[3].size(), 3);
   
   EXPECT_EQ(combinations[0][0], 1);
   EXPECT_EQ(combinations[0][1], 2);
   EXPECT_EQ(combinations[0][2], 3);

   EXPECT_EQ(combinations[1][0], 1);
   EXPECT_EQ(combinations[1][1], 2);
   EXPECT_EQ(combinations[1][2], 4);
   
   EXPECT_EQ(combinations[2][0], 1);
   EXPECT_EQ(combinations[2][1], 3);
   EXPECT_EQ(combinations[2][2], 4);

   EXPECT_EQ(combinations[3][0], 2);
   EXPECT_EQ(combinations[3][1], 3);
   EXPECT_EQ(combinations[3][2], 4);
   
   combinations = utility::GenerateAllCombinations(seed, 4);
   EXPECT_EQ(combinations.size()   , 1);
   EXPECT_EQ(combinations[0].size(), 4);
   EXPECT_EQ(combinations[0][0], 1);
   EXPECT_EQ(combinations[0][1], 2);
   EXPECT_EQ(combinations[0][2], 3);
   EXPECT_EQ(combinations[0][3], 4);

   EXPECT_THROW(utility::GenerateAllCombinations(seed, 5 );, std::runtime_error);
   EXPECT_THROW(utility::GenerateAllCombinations(seed, -1);, std::runtime_error);

}



} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_UTILITY_INTEGER_HPP_ */
