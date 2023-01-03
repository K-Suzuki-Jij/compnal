//
//  Copyright 2022 Kohei Suzuki
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
//  matrix_vector_product.hpp
//  compnal
//
//  Created by kohei on 2023/01/03.
//  
//

#ifndef COMPNAL_BLAS_MATRIX_VECTOR_PRODUCT_HPP_
#define COMPNAL_BLAS_MATRIX_VECTOR_PRODUCT_HPP_

namespace compnal {
namespace blas {

//! @brief Calculate matrix vector product. \f$ \boldsymbol{v}_{\rm out} = c\hat{M}\cdot\boldsymbol{v} \f$.
//! @tparam T1 The value type of the coefficient \f$ c\f$.
//! @tparam T2 The value type of the matirx \f$ \hat{M}\f$.
//! @tparam T3 The value type of the vector \f$ \boldsymbol{v}\f$.
//! @tparam T4 The value type of output vector.
//! @param coeff The coefficient \f$ c\f$.
//! @param matrix_in The matrix \f$ \hat{M} \f$.
//! @param vector_in The vector \f$ \boldsymbol{v} \f$.
//! @param vector_out The result of matrix vector product \f$ \boldsymbol{v}_{\rm out} = c\hat{M}\cdot\boldsymbol{v}\f$.
template<typename T1, typename T2, typename T3, typename T4>
void CalculateMatrixVectorProduct(const T1 coeff,
                                  const CRS<T2> &matrix_in,
                                  const std::vector<T3> &vector_in,
                                  std::vector<T4> *vector_out,
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
   using T2T3 = decltype(std::declval<T2>() * std::declval<T3>());

#pragma omp parallel for schedule(guided) num_threads(num_threads)
   for (std::int64_t i = 0; i < matrix_in.row_dim; ++i) {
      T2T3 temp = 0;
      for (std::int64_t j = matrix_in.row[i]; j < matrix_in.row[i + 1]; ++j) {
         temp += matrix_in.val[j] * vector_in[matrix_in.col[j]];
      }
      (*vector_out)[i] = temp*coeff;
   }
   
}

} // namespace blas
} // namespace compnal

#endif /* COMPNAL_BLAS_MATRIX_VECTOR_PRODUCT_HPP_ */
