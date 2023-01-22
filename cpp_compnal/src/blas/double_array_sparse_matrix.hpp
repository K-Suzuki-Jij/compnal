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
struct DASPM {
   
   DASPM(const std::int64_t dim = 0) {
      this->col_val.resize(dim);
      this->row_dim = dim;
      this->col_dim = dim;
   }
   
   std::unique_ptr<RealType[]> ToArray() const {
      std::unique_ptr<RealType[]> matrix_array = std::make_unique<RealType[]>(this->row_dim*this->col_dim);
      for (std::int64_t i = 0; i < row_dim; ++i) {
         for (std::size_t j = 0; j < this->col_val[i].size(); ++j) {
            matrix_array[i*this->col_dim + this->col_val[i][j].first] = this->col_val[i][j].second;
            matrix_array[this->col_val[i][j].first*this->col_dim + i] = this->col_val[i][j].second;
         }
      }
      return matrix_array;
   }
   
   std::int64_t row_dim = 0;
   std::int64_t col_dim = 0;
   std::vector<std::vector<std::pair<std::int64_t, RealType>>> col_val;
   
};


template<typename T1, typename T2, typename T3, typename T4>
void CalculateMatrixVectorProduct(std::vector<T1> *vector_out,
                                  const T2 coeff,
                                  const DASPM<T3> &matrix_in,
                                  const std::vector<T4> &vector_in,
                                  const std::int32_t num_threads = 1) {
   
   if (matrix_in.col_dim != vector_in.size()) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "The column of the input matrix is " << matrix_in.col_dim << std::endl;
      ss << "The dimension of the input vector is " << vector_in.size() << std::endl;
      ss << "Both must be equal" << std::endl;
      throw std::runtime_error(ss.str());
   }
   vector_out->resize(matrix_in.row_dim);
   using T3T4 = decltype(std::declval<T3>()*std::declval<T4>());
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < matrix_in.row_dim; ++i) {
      T3T4 temp = 0;
      for (std::size_t j = 0; j < matrix_in.col_val[i].size(); ++j) {
         temp += vector_in[matrix_in.col_val[i][j].first]*matrix_in.col_val[i][j].second;
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
   
   if (matrix_in.row_dim != matrix_in.col_dim) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "The input matrix is not symmetric" << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   if (matrix_in.col_dim != vector_in.size()) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "The column of the input matrix is " << matrix_in.col_dim << std::endl;
      ss << "The dimension of the input vector is " << vector_in.size() << std::endl;
      ss << "Both must be equal" << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   vector_out->resize(matrix_in.row_dim);
   
   if (static_cast<std::int32_t>(vectors_work->size()) != num_threads) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "Working vector (vectors_work) must be arrays of the number of parallel threads";
      ss << "The size of working vector is " << vectors_work->size();
      ss << "The number of parallel threads is num_threads";
      throw std::runtime_error(ss.str());
   }
   
#pragma omp parallel num_threads(num_threads)
   {
      const std::int32_t thread_num = omp_get_thread_num();
#pragma omp for schedule(guided)
      for (std::int64_t i = 0; i < matrix_in.row_dim; ++i) {
         const T4 temp_vec_in = vector_in[i];
         T1 temp_val = 0;
         std::int64_t col_size = static_cast<std::int64_t>(matrix_in.col_val[i].size()) - 1;
         for (std::int64_t j = 0; j < col_size; ++j) {
            temp_val += matrix_in.col_val[i][j].second*vector_in[matrix_in.col_val[i][j].first];
            (*vectors_work)[thread_num][matrix_in.col_val[i][j].first] += matrix_in.col_val[i][j].second*temp_vec_in;
         }
         (*vectors_work)[thread_num][i] += temp_val + matrix_in.col_val[i][col_size].second*temp_vec_in;
      }
   }
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < matrix_in.row_dim; ++i) {
      T1 temp_val = 0;
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
   
   if (matrix_in->row_dim != matrix_in->col_dim) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "The matrix is not a square matrix." << std::endl;
      throw std::runtime_error(ss.str());
   }
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < matrix_in->row_dim; ++i) {
      if (matrix_in->col_val[i].back().first != i) {
         throw std::runtime_error("Could not add diagonal elements");
      }
      matrix_in->col_val[i].back().second += diag_add;
   }
   
}

template<typename T1, typename T2>
void AddDiagonalElements(DASPM<T1> *matrix_in,
                         const T2 diag_add,
                         const std::int32_t num_threads = 1) {
   
   if (matrix_in->row_dim != matrix_in->col_dim) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "The matrix is not a square matrix." << std::endl;
      throw std::runtime_error(ss.str());
   }
   
#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < matrix_in->row_dim; ++i) {
      for (std::size_t j = 0; j < matrix_in->col_val[i].size(); ++j) {
         if (matrix_in->col_val[i][j].first == i) {
            matrix_in->col_val[i][j].second += diag_add;
            break;
         }
      }
   }
   
}

}
}


#endif /* double_array_sparse_matrix_h */
