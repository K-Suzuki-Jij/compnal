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
//  conjugate_gradient.hpp
//  compnal
//
//  Created by kohei on 2023/01/15.
//  
//

#ifndef COMPNAL_BLAS_CONJUGATE_GRADIENT_HPP_
#define COMPNAL_BLAS_CONJUGATE_GRADIENT_HPP_

#include "../utility/all.hpp"
#include "compressed_row_storage.hpp"
#include "matrix_vector_product.hpp"
#include "orthonormalize.hpp"
#include <omp.h>

namespace compnal {
namespace blas {

template <typename RealType>
struct CGParams {
   std::int32_t max_step = 1000;
   std::int32_t num_threads = 1;
   std::uint32_t seed = std::random_device()();
   RealType acc = std::pow(10, -7);
   bool flag_use_initial_vec = false;
   bool flag_symmetric_crs = false;
};

template<typename RealType>
void ConjugateGradient(std::vector<RealType> *vec_out,
                       const CRS<RealType> &matrix_in,
                       const std::vector<RealType> &vec_in,
                       const std::vector<std::vector<RealType>> &subspace_vectors = {},
                       const CGParams<RealType> &params = CGParams<RealType>()) {
   
   if (matrix_in.row_dim != matrix_in.col_dim) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "The input matrix is not a square one" << std::endl;
      ss << "row=" << matrix_in.row_dim << ", col=" << matrix_in.col_dim << std::endl;
      throw std::runtime_error(ss.str());
   }

   if (static_cast<std::int64_t>(vec_in.size()) != matrix_in.row_dim) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "Matrix vector product (Ax=b) cannot be defined." << std::endl;
      throw std::runtime_error(ss.str());
   }

   const std::int64_t dim = matrix_in.row_dim;
   std::vector<RealType> rrr(dim);
   std::vector<RealType> ppp(dim);
   std::vector<RealType> yyy(dim);
   std::vector<std::vector<RealType>> vectors_work;

   if (params.flag_symmetric_crs) {
      vectors_work = std::vector<std::vector<RealType>>(params.num_threads, std::vector<RealType>(dim));
   }

   if (params.flag_use_initial_vec) {
      if (static_cast<std::int64_t>(vec_out->size()) != dim) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The dimension of the initial vector is not equal to that of the input matrix." << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   else {
      std::uniform_real_distribution<RealType> uniform_rand(-1, 1);
      utility::RandType random_number_engine;
      random_number_engine.seed(params.seed);
      vec_out->resize(dim);
      for (std::int64_t i = 0; i < dim; ++i) {
         (*vec_out)[i] = uniform_rand(random_number_engine);
      }
   }
   Orthonormalize(vec_out, subspace_vectors, params.num_threads);

   if (params.flag_symmetric_crs) {
      CalculateSymmetricMatrixVectorProduct(&rrr, &vectors_work, 1, matrix_in, *vec_out, params.num_threads);
   }
   else {
      CalculateMatrixVectorProduct(&rrr, 1, matrix_in, *vec_out, params.num_threads);
   }

#pragma omp parallel for num_threads(params.num_threads)
   for (std::int64_t i = 0; i < dim; ++i) {
      rrr[i] = vec_in[i] - rrr[i];
      ppp[i] = rrr[i];
   }

   Orthonormalize(&rrr, subspace_vectors, params.num_threads, false);
   Orthonormalize(&ppp, subspace_vectors, params.num_threads, false);

   for (std::int32_t step = 0; step < params.max_step; ++step) {
      if (params.flag_symmetric_crs) {
         CalculateSymmetricMatrixVectorProduct(&yyy, &vectors_work, 1, matrix_in, ppp, params.num_threads);
      }
      else {
         CalculateMatrixVectorProduct(&yyy, 1, matrix_in, ppp, params.num_threads);
      }

      Orthonormalize(&yyy, subspace_vectors, params.num_threads, false);

      const RealType inner_prod = CalculateInnerProduct(rrr, rrr, params.num_threads);
      const RealType alpha = inner_prod/CalculateInnerProduct(ppp, yyy, params.num_threads);

#pragma omp parallel for num_threads(params.num_threads)
      for (std::int64_t i = 0; i < dim; ++i) {
         (*vec_out)[i] += alpha*ppp[i];
         rrr[i] -= alpha*yyy[i];
      }

      Orthonormalize(vec_out, subspace_vectors, params.num_threads, false);
      Orthonormalize(&rrr, subspace_vectors, params.num_threads, false);

      const RealType residual_error = CalculateInnerProduct(rrr, rrr, params.num_threads);

      if (residual_error < params.acc) {
         return;
      }

      const RealType beta = residual_error/inner_prod;

#pragma omp parallel for num_threads(params.num_threads)
      for (std::int64_t i = 0; i < dim; ++i) {
         ppp[i] = rrr[i] + beta*ppp[i];
      }
      
      Orthonormalize(&ppp, subspace_vectors, params.num_threads, false);
   }

   std::stringstream ss;
   ss << "Error in " << __func__ << std::endl;
   ss << "Does not converge" << std::endl;
   throw std::runtime_error(ss.str());
}

}  // namespace blas
}  // namespace compnal

#endif /* conjugate_gradient_h */
