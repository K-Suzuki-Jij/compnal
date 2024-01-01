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
   
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 1, 0.0, 2).first, 2.5 );
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 1, 3.0, 2).first, -0.5);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 2, 0.0, 2).first, 7.25);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 2, 3.0, 2).first, 1.25);

   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 1, 0.0, 2).second, 1.0);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 1, 3.0, 2).second, 1.0);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 2, 0.0, 2).second, 5.0);
   EXPECT_DOUBLE_EQ(utility::CalculateMomentWithSTD(samples, 2, 3.0, 2).second, 1.0);
   
}

TEST(Utility, CalculateFFTMagnitudeList) {
   
   Eigen::Matrix<double, 2, 3, Eigen::RowMajor> samples{
      {1, 2, 3},
      {4, 5, 6}
   };

   EXPECT_THROW(utility::CalculateFFTMagnitudeList(samples, 2, "ortho", 2), std::invalid_argument);
   EXPECT_THROW(utility::CalculateFFTMagnitudeList(samples, 3, "aaa", 2), std::invalid_argument);

   
   auto magnitude_list = utility::CalculateFFTMagnitudeList(samples, 3, "ortho", 2);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 0), 3.4641016151377553);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 1), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 2), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 0), 8.660254037844387);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 1), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 2), 1.0);

   magnitude_list = utility::CalculateFFTMagnitudeList(samples, 3, "forward", 2);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 0), 2.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 1), 0.5773502691896257);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 2), 0.5773502691896257);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 0), 5.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 1), 0.5773502691896257);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 2), 0.5773502691896257);

   magnitude_list = utility::CalculateFFTMagnitudeList(samples, 3, "backward", 2);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 0), 6.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 1), 1.7320508075688772);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 2), 1.7320508075688772);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 0), 15.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 1), 1.7320508075688772);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 2), 1.7320508075688772);

}

TEST(Utility, CalculateFFT2MagnitudeList) {
   
   Eigen::Matrix<double, 2, 4, Eigen::RowMajor> samples{
      {1, 2, 3, 4},
      {5, 6, 7, 8}
   };

   EXPECT_THROW(utility::CalculateFFT2MagnitudeList(samples, 2, 3, "ortho", 2), std::invalid_argument);
   EXPECT_THROW(utility::CalculateFFT2MagnitudeList(samples, 2, 2, "aaa", 2), std::invalid_argument);
   
   auto magnitude_list = utility::CalculateFFT2MagnitudeList(samples, 2, 2, "ortho", 2);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 0), 5.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 1), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 2), 2.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 3), 0.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 0), 13.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 1), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 2), 2.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 3), 0.0);

   magnitude_list = utility::CalculateFFT2MagnitudeList(samples, 2, 2, "forward", 2);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 0), 2.5);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 1), 0.5);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 2), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 3), 0.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 0), 6.5);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 1), 0.5);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 2), 1.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 3), 0.0);

   magnitude_list = utility::CalculateFFT2MagnitudeList(samples, 2, 2, "backward", 2);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 0), 10.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 1), 2.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 2), 4.0);
   EXPECT_DOUBLE_EQ(magnitude_list(0, 3), 0.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 0), 26.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 1), 2.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 2), 4.0);
   EXPECT_DOUBLE_EQ(magnitude_list(1, 3), 0.0);
}

}  // namespace test
}  // namespace compnal
