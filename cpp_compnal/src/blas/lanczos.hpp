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
//  lanczos.hpp
//  compnal
//
//  Created by kohei on 2023/01/14.
//  
//

#ifndef COMPNAL_BLAS_LANCZOS_HPP_
#define COMPNAL_BLAS_LANCZOS_HPP_

#include "../utility/all.hpp"
#include "orthonormalize.hpp"
#include "lapack_wrapper.hpp"
#include "compressed_row_storage.hpp"
#include "double_array_sparse_matrix.hpp"
#include <omp.h>
#include <chrono>

namespace compnal {
namespace blas {

template<typename RealType>
struct DiagParams {
   std::int32_t min_step = 0;
   std::int32_t max_step = 1000;
   std::int32_t num_threads = 1;
   std::uint32_t seed = std::random_device()();
   RealType acc = std::pow(10, -14);
   bool flag_use_initial_vec = false;
   bool flag_store_vec = false;
   bool flag_symmetric_crs = false;
   bool flag_display_info = false;
};


template<typename RealType, class SPMType>
void EigendecompositionLanczos(RealType *target_value_out,
                               std::vector<RealType> *target_vector_out,
                               const SPMType &matrix_in,
                               const std::vector<std::vector<RealType>> &subspace_vectors = {},
                               const DiagParams<RealType> params = DiagParams<RealType>()
                               ) {
   
   const auto start = std::chrono::system_clock::now();
   std::ios::fmtflags flagsSaved = std::cout.flags();
   /*
   if (matrix_in.row_dim != matrix_in.col_dim) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "The input matrix is not a square one" << std::endl;
      ss << "row=" << matrix_in.row_dim << ", col=" << matrix_in.col_dim << std::endl;
      throw std::runtime_error(ss.str());
   }

   if (matrix_in.row_dim == 0) {
      *target_value_out = 0.0;
      target_vector_out->clear();
      return;
   }

   if (matrix_in.row_dim <= 1000) {
      const std::int32_t dim_subspace = static_cast<std::int32_t>(subspace_vectors.size());
      LapackSYEV<RealType>(target_value_out, target_vector_out, dim_subspace, matrix_in);
      return;
   }
    */
   
   if (params.max_step <= 0) {
      return;
   }

   std::int32_t converge_step_number = 0;
   //const std::int64_t dim = matrix_in.row_dim;
   const std::int64_t dim = matrix_in.size();
   RealType residual_error_final = 0.0;
   std::vector<RealType> vector_0(dim);
   std::vector<RealType> vector_1(dim);
   std::vector<RealType> vector_2(dim);

   std::vector<std::vector<RealType>> rits_vector;
   std::vector<RealType> krylov_eigen_vector;
   std::vector<RealType> krylov_eigen_value(params.max_step + 1);
   std::vector<RealType> diagonal_value;
   std::vector<RealType> off_diagonal_value;
   std::vector<std::vector<RealType>> vectors_work;

   std::uniform_real_distribution<RealType> uniform_rand(-1, 1);
   utility::RandType random_number_engine;
   
   if (params.flag_symmetric_crs) {
      vectors_work = std::vector<std::vector<RealType>>(params.num_threads, std::vector<RealType>(dim));
   }

   if (params.flag_use_initial_vec) {
      if (target_vector_out->size() != vector_0.size()) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "Cannot initialize initial state" << std::endl;
         ss << "The size of initial state=" << target_vector_out->size();
         ss << ", the dimention=" << vector_0.size() << std::endl;
         throw std::runtime_error(ss.str());
      }
      CopyVector(&vector_0, *target_vector_out);
   }
   else {
      random_number_engine.seed(params.seed);
      for (std::int64_t i = 0; i < dim; ++i) {
         vector_0[i] = uniform_rand(random_number_engine);
      }
   }

   Orthonormalize(&vector_0, subspace_vectors, params.num_threads);

   if (params.flag_store_vec) {
      rits_vector.push_back(vector_0);
   }

   if (params.flag_symmetric_crs) {
      CalculateSymmetricMatrixVectorProduct(&vector_1, &vectors_work, 1, matrix_in, vector_0, params.num_threads);
   }
   else {
      CalculateMatrixVectorProduct(&vector_1, 1, matrix_in, vector_0, params.num_threads);
   }

   diagonal_value.push_back(CalculateInnerProduct(vector_0, vector_1, params.num_threads));
   krylov_eigen_value[0] = diagonal_value[0];
   CalculateVectorVectorSum(&vector_1, 1, vector_1, -krylov_eigen_value[0], vector_0, params.num_threads);

   for (std::int32_t step = 0; step < params.max_step; ++step) {
      CopyVector(&vector_2, vector_1, params.num_threads);
      off_diagonal_value.push_back(CalculateL2Norm(vector_2, params.num_threads));
      Orthonormalize(&vector_2, subspace_vectors, params.num_threads);

      if (params.flag_store_vec) {
         rits_vector.push_back(vector_2);
      }

      if (params.flag_symmetric_crs) {
         CalculateSymmetricMatrixVectorProduct(&vector_1, &vectors_work, 1, matrix_in, vector_2, params.num_threads);
      }
      else {
         CalculateMatrixVectorProduct(&vector_1, 1, matrix_in, vector_2, params.num_threads);
      }

      diagonal_value.push_back(CalculateInnerProduct(vector_1, vector_2, params.num_threads));

      if (step >= params.min_step) {
         LapackSTEV<RealType>(&krylov_eigen_value[step + 1], &krylov_eigen_vector, diagonal_value, off_diagonal_value);
         
         const RealType residual_error = std::abs(krylov_eigen_value[step + 1] - krylov_eigen_value[step]);
         if (params.flag_display_info) {
            std::cout << "\rLanczos Step[" << step + 1 << "]=" << std::scientific << std::setprecision(1);
            std::cout << residual_error << std::flush;
         }
         if (residual_error < params.acc) {
            *target_value_out = krylov_eigen_value[step + 1];
            converge_step_number = step + 1;
            residual_error_final = residual_error;
            break;
         }
      }
      
#pragma omp parallel for num_threads(params.num_threads)
      for (std::int64_t i = 0; i < dim; ++i) {
         vector_1[i] -= diagonal_value[step + 1]*vector_2[i] + off_diagonal_value[step]*vector_0[i];
         vector_0[i] = vector_2[i];
      }
   }

   if (converge_step_number <= 0) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "Did not converge" << std::endl;
      throw std::runtime_error(ss.str());
   }

   target_vector_out->resize(dim);

   if (params.flag_store_vec) {
#pragma omp parallel for num_threads(params.num_threads)
      for (std::int64_t i = 0; i < dim; ++i) {
         RealType temp_val = 0.0;
         for (std::int32_t j = 0; j <= converge_step_number; ++j) {
            temp_val += krylov_eigen_vector[j]*rits_vector[j][i];
         }
         (*target_vector_out)[i] = temp_val;
      }
      Normalize(target_vector_out);
   }
   else {
      if (params.flag_use_initial_vec) {
#pragma omp parallel for num_threads(params.num_threads)
         for (std::int64_t i = 0; i < dim; ++i) {
            vector_0[i] = (*target_vector_out)[i];
            (*target_vector_out)[i] = 0.0;
         }
      }
      else {
         random_number_engine.seed(params.seed);
         for (std::int64_t i = 0; i < dim; ++i) {
            vector_0[i] = uniform_rand(random_number_engine);
         }
      }

      Orthonormalize(&vector_0, subspace_vectors, params.num_threads);

      CalculateVectorVectorSum(target_vector_out, 1, *target_vector_out, krylov_eigen_vector[0], vector_0, params.num_threads);
      
      if (params.flag_symmetric_crs) {
         CalculateSymmetricMatrixVectorProduct(&vector_1, &vectors_work, 1, matrix_in, vector_0, params.num_threads);
      }
      else {
         CalculateMatrixVectorProduct(&vector_1, 1, matrix_in, vector_0, params.num_threads);
      }

      CalculateVectorVectorSum(&vector_1, 1, vector_1, -krylov_eigen_value[0], vector_0, params.num_threads);

      for (std::int32_t step = 1; step <= converge_step_number; ++step) {
         CopyVector(&vector_2, vector_1, params.num_threads);
         Orthonormalize(&vector_2, subspace_vectors, params.num_threads);
         CalculateVectorVectorSum(target_vector_out, 1, *target_vector_out, krylov_eigen_vector[step], vector_2, params.num_threads);

         if (params.flag_symmetric_crs) {
            CalculateSymmetricMatrixVectorProduct(&vector_1, &vectors_work, 1, matrix_in, vector_2, params.num_threads);
         }
         else {
            CalculateMatrixVectorProduct(&vector_1, 1, matrix_in, vector_2, params.num_threads);
         }

#pragma omp parallel for num_threads(params.num_threads)
         for (std::int64_t i = 0; i < dim; ++i) {
            vector_1[i] -= diagonal_value[step]*vector_2[i] + off_diagonal_value[step - 1]*vector_0[i];
            vector_0[i] = vector_2[i];
         }
         if (params.flag_display_info) {
            std::cout << "\rLanczos Vec Step:" << step << "/";
            std::cout << converge_step_number << std::string(5, ' ') << std::flush;
         }
      }
      Normalize(target_vector_out);
   }
   
   if (params.flag_display_info) {
      const std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
      std::cout << std::defaultfloat << std::fixed << std::setprecision(3);
      std::cout << "\rDiagonalize by Lanczos:";
      std::cout << elapsed_seconds.count() << "[sec]" << std::flush;
      std::cout << std::scientific << std::setprecision(1);
      std::cout << " (" << residual_error_final << ")" << std::flush;
      std::cout << std::string(30, ' ') << std::flush;
      std::cout.flags(flagsSaved);
   }
    

}



}  // namespace blas
}  // namespace compnal


#endif /* COMPNAL_BLAS_LANCZOS_HPP_ */
