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
//  inverse_iteration.hpp
//  compnal
//
//  Created by kohei on 2023/01/15.
//  
//

#ifndef COMPNAL_BLAS_INVERSE_ITERATION_HPP_
#define COMPNAL_BLAS_INVERSE_ITERATION_HPP_

#include "compressed_row_storage.hpp"
#include "conjugate_gradient.hpp"
#include <omp.h>

namespace compnal {
namespace blas {

template<typename RealType>
struct IIParams {
   std::int32_t max_step = 3;
   std::int32_t num_threads = 1;
   RealType diag_add = std::pow(10, -11);
   RealType acc = std::pow(10, -7);
   LinearEqParams<RealType> linear_eq_params;
   LinearEqAlgorithm linear_eq_method = LinearEqAlgorithm::CONJUGATE_GRADIENT;
};

template<typename RealType>
void InverseIteration(CRS<RealType> *matrix_in,
                      std::vector<RealType> *eigenvector,
                      const RealType eigenvalue,
                      const std::vector<std::vector<RealType>> &subspace_vectors = {},
                      const IIParams<RealType> &params = IIParams<RealType>()) {
   
   if (matrix_in->row_dim != matrix_in->col_dim) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "The input matrix is not a square one" << std::endl;
      ss << "row=" << matrix_in->row_dim << ", col=" << matrix_in->col_dim << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   if (params.max_step <= 0) {
      return;
   }

   std::vector<RealType> improved_eigenvector;
   std::vector<RealType> vectors_work(matrix_in->row_dim);
   std::vector<std::vector<RealType>> vectors_work_pthreads;

   if (params.linear_eq_params.flag_use_initial_vec) {
      if (static_cast<std::int64_t>(eigenvector->size()) != matrix_in->row_dim) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The dimension of the initial vector is not equal to that of the input matrix." << std::endl;
         throw std::runtime_error(ss.str());
      }
      improved_eigenvector = *eigenvector;
   }

   matrix_in->AddDiagonalElements(params.diag_add - eigenvalue);

   for (std::int32_t step = 0; step < params.max_step; ++step) {
      if (params.linear_eq_params.flag_symmetric_crs) {
         std::vector<std::vector<RealType>> vectors_work_pthreads(params.num_threads, std::vector<RealType>(matrix_in->row_dim));
         CalculateSymmetricMatrixVectorProduct(&vectors_work, &vectors_work_pthreads, 1, *matrix_in, *eigenvector, params.num_threads);
      }
      else {
         CalculateMatrixVectorProduct(&vectors_work, 1, *matrix_in, *eigenvector);
      }
      const RealType residual_error = CalculateL1Distance(params.diag_add, *eigenvector, 1, vectors_work, params.num_threads);
      printf("ii_error=%.15lf\n", residual_error);
      if (residual_error < params.acc) {
         matrix_in->AddDiagonalElements(-(params.diag_add - eigenvalue));
         return;
      }
      if (params.linear_eq_method == LinearEqAlgorithm::CONJUGATE_GRADIENT) {
         ConjugateGradient(&improved_eigenvector, *matrix_in, *eigenvector, subspace_vectors, params.linear_eq_params);
      }
      else if (params.linear_eq_method == LinearEqAlgorithm::MINIMUM_RESIDUAL) {
         //MinimumResidual(&improved_eigenvector, *matrix_in, *eigenvector, subspace_vectors, params.linear_eq_params);
      }
      else {
         throw std::runtime_error("Unknwon linear equation solver detected.");
      }
      Normalize(&improved_eigenvector);
      CopyVector(eigenvector, improved_eigenvector);
   }

   std::stringstream ss;
   ss << "Error in " << __func__ << std::endl;
   ss << "Did not converge" << std::endl;
   throw std::runtime_error(ss.str());
}



}  // namespace blas
}  // namespace compnal

#endif /* COMPNAL_BLAS_INVERSE_ITERATION_HPP_ */
