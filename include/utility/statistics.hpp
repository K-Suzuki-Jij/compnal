//
//  Copyright 2024 Kohei Suzuki
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
#include <unsupported/Eigen/FFT>

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

//! @brief Calculate the moment of samples and it's variance.
//! @param samples Samples.
//! @param order Order of moment.
//! @param bias The bias in E((X - bias)^order). Defaults to 0.0.
//! @param num_threads The Number of calculation threads. Defaults to 1.
//! @return The moment of samples and it's variance.
std::pair<double, double> CalculateMomentWithSTD(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &samples,
                                                 const std::int32_t order, const double bias = 0.0, const std::int32_t num_threads = 1) {
   
   const std::int64_t num_samples = samples.rows();
   double moment = 0.0;
   double std = 0.0;
   double squre_moment = 0.0;
   
#pragma omp parallel for schedule(guided) num_threads(num_threads) reduction(+: moment, squre_moment)
   for (std::int64_t i = 0; i < num_samples; ++i) {
      double mean = samples.row(i).mean();
      double biased_mean = 1.0;
      for (std::int32_t j = 0; j < order; ++j) {
         biased_mean *= mean - bias;
      }
      moment += biased_mean;
      squre_moment += biased_mean*biased_mean;
   }
   moment = moment/num_samples;
   std = std::sqrt(squre_moment/num_samples - moment*moment);
   
   return {moment, std};
}

//! @brief Calculates the 1D Fourier transform magnitude for each sample in a list of arrays.
//! @param array_list The input matrix where each row represents a 1D array.
//! @param n The number of elements in the 1D array.
//! @param norm The normalization type to use.
//! @param power The exponent to be applied to the calculated Fourier intensities.
//! @param num_threads The number of threads to use for parallel computation.
//! @return A matrix where each row contains the Fourier transform magnitude of the corresponding input array.
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
CalculateFFTMagnitudeList(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &array_list,
                          const std::int64_t n,
                          const std::string norm,
                          const std::int32_t power,
                          const std::int32_t num_threads) {
   
   if (array_list.cols() != n) {
      throw std::invalid_argument("The number of columns in array_list must be equal to n.");
   }
   
   std::int64_t num_samples = static_cast<std::int64_t>(array_list.rows());
   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fft_intensity(num_samples, n);
   
   // Set normalization factor
   double norm_factor = 1.0;
   if (norm == "ortho") {
      norm_factor = 1.0/std::sqrt(n);
   }
   else if (norm == "forward") {
      norm_factor = 1.0/n;
   }
   else if (norm == "backward") {
      norm_factor = 1.0;
   }
   else {
      throw std::invalid_argument("Invalid norm type: " + norm);
   }
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < num_samples; ++i) {
      Eigen::FFT<double> fft;
      Eigen::VectorXcd fft_result(n);
      
      // Applying FFT
      const Eigen::VectorXd &new_row = array_list.row(i);
      fft.fwd(fft_result, new_row);
      
      // Calculating Fourier intensity
      if (power == 1) {
         fft_intensity.row(i) = fft_result.array().abs()*norm_factor;
      }
      else {
         fft_intensity.row(i) = (fft_result.array().abs()*norm_factor).pow(power);
      }

   }
   
   return fft_intensity;
}


//! @brief Calculates the 2D Fourier transform magnitude for each sample in a list of arrays.
//! @param array_list The input matrix where each row represents a flattened 2D array.
//! @param n_x The number of columns in the unflattened 2D array.
//! @param n_y The number of rows in the unflattened 2D array.
//! @param norm The normalization type to use.
//! @param power The exponent to be applied to the calculated Fourier intensities.
//! @param num_threads The number of threads to use for parallel computation.
//! @return A matrix where each row contains the Fourier transform magnitude of the corresponding input array.
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
CalculateFFT2MagnitudeList(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &array_list,
                           const std::int64_t n_x,
                           const std::int64_t n_y,
                           const std::string norm,
                           const double power,
                           const std::int32_t num_threads) {
   if (array_list.cols() != n_x*n_y) {
      throw std::invalid_argument("The number of columns in array_list must be equal to n_x*n_y.");
   }
   
   std::int64_t num_samples = static_cast<std::int64_t>(array_list.rows());
   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fft_intensity(num_samples, n_x*n_y);
   
   //Set normalization factor
   double norm_factor = 1.0;
   if (norm == "ortho") {
      norm_factor = 1.0/std::sqrt(n_x*n_y);
   }
   else if (norm == "forward") {
      norm_factor = 1.0/(n_x*n_y);
   }
   else if (norm == "backward") {
      norm_factor = 1.0;
   }
   else {
      throw std::invalid_argument("Invalid norm type: " + norm);
   }
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < num_samples; ++i) {
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp_fft_matrix(n_x, n_y);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp_matrix = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(array_list.row(i).data(), n_x, n_y);
      Eigen::FFT<double> fft;
      
      // Applying FFT to each row
      Eigen::VectorXcd fft_row(n_y);
      for (std::int64_t j = 0; j < n_x; ++j) {
         fft.fwd(fft_row, temp_matrix.row(j)); // Forward FFT
         temp_fft_matrix.row(j) = fft_row; // Storing the result
      }
      
      // Applying FFT to each column
      Eigen::VectorXcd fft_col(n_x);
      for (std::int64_t j = 0; j < n_y; ++j) {
         fft.fwd(fft_col, temp_fft_matrix.col(j)); // Forward FFT
         temp_fft_matrix.col(j) = fft_col; // Storing the result
      }
      
      // Calculating Fourier intensity
      if (power == 1) {
         fft_intensity.row(i) = temp_fft_matrix.array().abs().reshaped<Eigen::RowMajor>()*norm_factor;
      }
      else {
         fft_intensity.row(i) = (temp_fft_matrix.array().abs()*norm_factor).pow(power).reshaped<Eigen::RowMajor>();
      }
   }
   
   return fft_intensity;
}


} // namespace utility
} // namespac

