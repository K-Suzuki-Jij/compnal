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
//  double_array_sparse_matrix.hpp
//  compnal
//
//  Created by kohei on 2023/01/21.
//  
//

#ifndef COMPNAL_BLAS_DOUBLE_ARRAY_SPARSE_MATRIX_HPP_
#define COMPNAL_BLAS_DOUBLE_ARRAY_SPARSE_MATRIX_HPP_

namespace compnal {
namespace blas {

template<typename RealType>
using DASPM = std::vector<std::vector<std::pair<std::int64_t, RealType>>>;

template<typename T1, typename T2, typename T3, typename T4>
void CalculateMatrixVectorProduct(std::vector<T1> *vector_out,
                                  const T2 coeff,
                                  const DASPM<T3> &matrix_in,
                                  const std::vector<T4> &vector_in,
                                  const std::int32_t num_threads = 1) {
   
   vector_out->resize(matrix_in.size());
   using T3T4 = decltype(std::declval<T3>()*std::declval<T4>());
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::size_t i = 0; i < matrix_in.size(); ++i) {
      T3T4 temp = 0;
      for (std::size_t j = 0; j < matrix_in[i].size(); ++j) {
         temp += vector_in[matrix_in[i][j].first]*matrix_in[i][j].second;
      }
      (*vector_out)[i] = temp*coeff;
   }
   
}

template<typename T1, typename T2, typename T3, typename T4>
void CalculateSymmetricMatrixVectorProduct(std::vector<T1> *vector_out,
                                           std::vector<std::vector<T1>> *vectors_work,
                                           const T2 coeff,
                                           const DASPM<T3> &matrix_in,
                                           const std::vector<T4> &vector_in,
                                           const std::int32_t num_threads = 1) {
   
   vector_out->resize(matrix_in.size());
   using T3T4 = decltype(std::declval<T3>()*std::declval<T4>());
   
#pragma omp parallel num_threads(num_threads)
   {
      const std::int32_t thread_num = omp_get_thread_num();
#pragma omp for schedule(guided)
      for (std::size_t i = 0; i < matrix_in.size(); ++i) {
         const T4 temp_vec_in = vector_in[i];
         T1 temp_val = matrix_in[i].back().second*temp_vec_in;
         for (std::int64_t j = 0; j < static_cast<std::int64_t>(matrix_in[i].size()) - 1; ++j) {
            temp_val += matrix_in[i][j].second*vector_in[matrix_in[i][j].first];
            (*vectors_work)[thread_num][matrix_in[i][j].first] += matrix_in[i][j].second*temp_vec_in;
         }
         (*vectors_work)[thread_num][i] += temp_val;
      }
   }
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::size_t i = 0; i < matrix_in.size(); ++i) {
      T1 temp_val = 0.0;
      for (std::int32_t thread_num = 0; thread_num < num_threads; ++thread_num) {
         temp_val += (*vectors_work)[thread_num][i];
         (*vectors_work)[thread_num][i] = 0;
      }
      (*vector_out)[i] = temp_val*coeff;
   }
}

template<typename T1, typename T2>
void AddSymmetricDiagonalElements(DASPM<T1> *matrix_in,
                                  const T2 diag_add,
                                  const std::int32_t num_threads = 1) {
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::size_t i = 0; i < matrix_in->size(); ++i) {
      if ((*matrix_in)[i].back().first != i) {
         throw std::runtime_error("Could not add diagonal elements");
      }
      (*matrix_in)[i].back().second += diag_add;
   }
   
}

template<typename T1, typename T2>
void AddDiagonalElements(DASPM<T1> *matrix_in,
                         const T2 diag_add,
                         const std::int32_t num_threads = 1) {
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::size_t i = 0; i < matrix_in->size(); ++i) {
      for (std::size_t j = 0; j < (*matrix_in)[i].size(); ++j) {
         if ((*matrix_in)[i][j].first == i) {
            (*matrix_in)[i][j].second += diag_add;
            break;
         }
      }
   }
   
}

}
}


#endif /* double_array_sparse_matrix_h */
