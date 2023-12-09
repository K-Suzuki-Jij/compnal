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
//  statistics.hpp
//  compnal
//
//  Created by kohei on 2023/07/25.
//  
//

#pragma once


#include <Eigen/Dense>

namespace compnal {
namespace utility {

//! @brief Calculate the moment of samples.
//! @param samples Samples.
//! @param order Order of moment.
//! @param bias The bias in E((X - bias)^order). Defaults to 0.0.
//! @param num_threads The Number of calculation threads. Defaults to 1.
//! @return The moment of samples.
double CalculateMoment(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &samples,
                       const std::int32_t order, const double bias = 0.0, const std::int32_t num_threads = 1) {
   
   const std::int64_t num_samples = samples.rows();
   double moment = 0.0;
   
#pragma omp parallel for schedule(guided) num_threads(num_threads) reduction(+: moment)
   for (std::int64_t i = 0; i < num_samples; ++i) {
      double mean = samples.row(i).mean();
      double biased_mean = 1.0;
      for (std::int32_t j = 0; j < order; ++j) {
         biased_mean *= mean - bias;
      }
      moment += biased_mean;
   }
      
   return moment/num_samples;
}

} // namespace utility
} // namespac

