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
//  combination.hpp
//  compnal
//
//  Created by kohei on 2023/06/29.
//  
//

#pragma once

#include "../../../include/utility/combination.hpp"

namespace compnal {
namespace test {


TEST(Utility, Combination) {
   std::vector<std::int32_t> vec = {1, 3, 5, 6, 7};
   std::vector<std::vector<std::int32_t>> ans_list;
   std::int32_t k = 1;
   
   do {
      std::vector<std::int32_t> ans;
      for (std::int32_t i = 0; i < k; ++i) {
         ans.push_back(vec[i]);
      }
      ans_list.push_back(ans);
   } while(utility::NextCombination(vec.begin(), vec.begin() + k, vec.end()));
   
   EXPECT_EQ(ans_list.at(0).at(0), 1);
   EXPECT_EQ(ans_list.at(1).at(0), 3);
   EXPECT_EQ(ans_list.at(2).at(0), 5);
   EXPECT_EQ(ans_list.at(3).at(0), 6);
   EXPECT_EQ(ans_list.at(4).at(0), 7);
   
   k = 2;
   ans_list.clear();
   do {
      std::vector<std::int32_t> ans;
      for (std::int32_t i = 0; i < k; ++i) {
         ans.push_back(vec[i]);
      }
      ans_list.push_back(ans);
   } while(utility::NextCombination(vec.begin(), vec.begin() + k, vec.end()));
   
   EXPECT_EQ(ans_list.at(0).at(0), 1);
   EXPECT_EQ(ans_list.at(0).at(1), 3);
   EXPECT_EQ(ans_list.at(1).at(0), 1);
   EXPECT_EQ(ans_list.at(1).at(1), 5);
   EXPECT_EQ(ans_list.at(2).at(0), 1);
   EXPECT_EQ(ans_list.at(2).at(1), 6);
   EXPECT_EQ(ans_list.at(3).at(0), 1);
   EXPECT_EQ(ans_list.at(3).at(1), 7);
   EXPECT_EQ(ans_list.at(4).at(0), 3);
   EXPECT_EQ(ans_list.at(4).at(1), 5);
   EXPECT_EQ(ans_list.at(5).at(0), 3);
   EXPECT_EQ(ans_list.at(5).at(1), 6);
   EXPECT_EQ(ans_list.at(6).at(0), 3);
   EXPECT_EQ(ans_list.at(6).at(1), 7);
   EXPECT_EQ(ans_list.at(7).at(0), 5);
   EXPECT_EQ(ans_list.at(7).at(1), 6);
   EXPECT_EQ(ans_list.at(8).at(0), 5);
   EXPECT_EQ(ans_list.at(8).at(1), 7);
   EXPECT_EQ(ans_list.at(9).at(0), 6);
   EXPECT_EQ(ans_list.at(9).at(1), 7);
   
   k = 3;
   ans_list.clear();
   do {
      std::vector<std::int32_t> ans;
      for (std::int32_t i = 0; i < k; ++i) {
         ans.push_back(vec[i]);
      }
      ans_list.push_back(ans);
   } while(utility::NextCombination(vec.begin(), vec.begin() + k, vec.end()));

   EXPECT_EQ(ans_list.at(0).at(0), 1);
   EXPECT_EQ(ans_list.at(0).at(1), 3);
   EXPECT_EQ(ans_list.at(0).at(2), 5);

   EXPECT_EQ(ans_list.at(1).at(0), 1);
   EXPECT_EQ(ans_list.at(1).at(1), 3);
   EXPECT_EQ(ans_list.at(1).at(2), 6);

   EXPECT_EQ(ans_list.at(2).at(0), 1);
   EXPECT_EQ(ans_list.at(2).at(1), 3);
   EXPECT_EQ(ans_list.at(2).at(2), 7);

   EXPECT_EQ(ans_list.at(3).at(0), 1);
   EXPECT_EQ(ans_list.at(3).at(1), 5);
   EXPECT_EQ(ans_list.at(3).at(2), 6);

   EXPECT_EQ(ans_list.at(4).at(0), 1);
   EXPECT_EQ(ans_list.at(4).at(1), 5);
   EXPECT_EQ(ans_list.at(4).at(2), 7);

   EXPECT_EQ(ans_list.at(5).at(0), 1);
   EXPECT_EQ(ans_list.at(5).at(1), 6);
   EXPECT_EQ(ans_list.at(5).at(2), 7);

   EXPECT_EQ(ans_list.at(6).at(0), 3);
   EXPECT_EQ(ans_list.at(6).at(1), 5);
   EXPECT_EQ(ans_list.at(6).at(2), 6);

   EXPECT_EQ(ans_list.at(7).at(0), 3);
   EXPECT_EQ(ans_list.at(7).at(1), 5);
   EXPECT_EQ(ans_list.at(7).at(2), 7);

   EXPECT_EQ(ans_list.at(8).at(0), 3);
   EXPECT_EQ(ans_list.at(8).at(1), 6);
   EXPECT_EQ(ans_list.at(8).at(2), 7);

   EXPECT_EQ(ans_list.at(9).at(0), 5);
   EXPECT_EQ(ans_list.at(9).at(1), 6);
   EXPECT_EQ(ans_list.at(9).at(2), 7);

   k = 4;
   ans_list.clear();
   do {
      std::vector<std::int32_t> ans;
      for (std::int32_t i = 0; i < k; ++i) {
         ans.push_back(vec[i]);
      }
      ans_list.push_back(ans);
   } while(utility::NextCombination(vec.begin(), vec.begin() + k, vec.end()));

   EXPECT_EQ(ans_list.at(0).at(0), 1);
   EXPECT_EQ(ans_list.at(0).at(1), 3);
   EXPECT_EQ(ans_list.at(0).at(2), 5);
   EXPECT_EQ(ans_list.at(0).at(3), 6);

   EXPECT_EQ(ans_list.at(1).at(0), 1);
   EXPECT_EQ(ans_list.at(1).at(1), 3);
   EXPECT_EQ(ans_list.at(1).at(2), 5);
   EXPECT_EQ(ans_list.at(1).at(3), 7);

   EXPECT_EQ(ans_list.at(2).at(0), 1);
   EXPECT_EQ(ans_list.at(2).at(1), 3);
   EXPECT_EQ(ans_list.at(2).at(2), 6);
   EXPECT_EQ(ans_list.at(2).at(3), 7);

   EXPECT_EQ(ans_list.at(3).at(0), 1);
   EXPECT_EQ(ans_list.at(3).at(1), 5);
   EXPECT_EQ(ans_list.at(3).at(2), 6);
   EXPECT_EQ(ans_list.at(3).at(3), 7);

   EXPECT_EQ(ans_list.at(4).at(0), 3);
   EXPECT_EQ(ans_list.at(4).at(1), 5);
   EXPECT_EQ(ans_list.at(4).at(2), 6);
   EXPECT_EQ(ans_list.at(4).at(3), 7);

   k = 5;
   ans_list.clear();
   do {
      std::vector<std::int32_t> ans;
      for (std::int32_t i = 0; i < k; ++i) {
         ans.push_back(vec[i]);
      }
      ans_list.push_back(ans);
   } while(utility::NextCombination(vec.begin(), vec.begin() + k, vec.end()));

   EXPECT_EQ(ans_list.at(0).at(0), 1);
   EXPECT_EQ(ans_list.at(0).at(1), 3);
   EXPECT_EQ(ans_list.at(0).at(2), 5);
   EXPECT_EQ(ans_list.at(0).at(3), 6);
   EXPECT_EQ(ans_list.at(0).at(4), 7);
}


}  // namespace test
}  // namespace compnal
