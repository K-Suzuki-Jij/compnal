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
//  test_statistics.hpp
//  compnal
//
//  Created by kohei on 2023/07/25.
//  
//

#pragma once

#include "../../../include/utility/statistics.hpp"

namespace compnal {
namespace test {

TEST(Utility, CalculateMoment) {
   
   Eigen::Matrix<double, 2, 2, Eigen::RowMajor> samples{
      {1, 2},
      {3, 4}
   };
   
   EXPECT_DOUBLE_EQ(utility::CalculateMoment(samples, 1, 0.0, 2), 2.5);
   EXPECT_DOUBLE_EQ(utility::CalculateMoment(samples, 1, 3.0, 2), -0.5);
   EXPECT_DOUBLE_EQ(utility::CalculateMoment(samples, 2, 0.0, 2), 7.25);
   EXPECT_DOUBLE_EQ(utility::CalculateMoment(samples, 2, 3.0, 2), 1.25);
   
}

TEST(Utility, CalculateMomentWithVariance) {
   
   Eigen::Matrix<double, 2, 2, Eigen::RowMajor> samples{
      {1, 2},
      {3, 4}
   };
   
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 1, 0.0, 2).first, 2.5 );
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 1, 3.0, 2).first, -0.5);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 2, 0.0, 2).first, 7.25);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 2, 3.0, 2).first, 1.25);

   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 1, 0.0, 2).second, 1.0 );
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 1, 3.0, 2).second, 1.0 );
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 2, 0.0, 2).second, 25.0);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithVariance(samples, 2, 3.0, 2).second, 1.0 );
   
}



}  // namespace test
}  // namespace compnal
