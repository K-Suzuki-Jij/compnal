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
//  vector_vector_operation.hpp
//  compnal
//
//  Created by kohei on 2023/01/15.
//  
//

#ifndef COMPNAL_BLAS_VECTOR_VECTOR_OPERATION_HPP_
#define COMPNAL_BLAS_VECTOR_VECTOR_OPERATION_HPP_

#include <cmath>

namespace compnal {
namespace blas {

template<typename RealType>
RealType CalculateInnerProduct(const std::vector<RealType> &v_1,
                               const std::vector<RealType> &v_2,
                               const std::int32_t num_threads = 1) {
   
   RealType inner_product = 0;
   
#pragma omp parallel for reduction(+: inner_product) num_threads(num_threads)
   for (std::size_t i = 0; i < v_1.size(); ++i) {
      inner_product += v_1[i]*v_2[i];
   }
   
   return inner_product;
   
}

template<typename RealType>
RealType CalculateL1Norm(const std::vector<RealType> &v,
                         const std::int32_t num_threads = 1) {
   
   RealType norm = 0;
   
#pragma omp parallel for reduction(+: norm) num_threads(num_threads)
   for (std::size_t i = 0; i < v.size(); ++i) {
      norm += std::abs(v[i]);
   }
   
   return norm;
   
}

template<typename RealType>
RealType CalculateL2Norm(const std::vector<RealType> &v,
                         const std::int32_t num_threads = 1) {
   
   return std::sqrt(CalculateInnerProduct(v, v, num_threads));
   
}

template<typename RealType>
void Normalize(std::vector<RealType> *v,
               const std::int32_t num_threads = 1) {
   
   const RealType norm = CalculateL2Norm(*v, num_threads);
   
   if (std::abs(norm) < std::numeric_limits<RealType>::epsilon()) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "Failed to orthonormalize." << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   const RealType coeff = 1/norm;
   
#pragma omp parallel for num_threads(num_threads)
   for (std::size_t i = 0; i < v->size(); ++i) {
      (*v)[i] = (*v)[i]*coeff;
   }
   
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
void CalculateVectorVectorSum(std::vector<T1> *vector_out,
                              const T2 coeff_1,
                              const std::vector<T3> &vector_1,
                              const T4 coeff_2,
                              const std::vector<T5> &vector_2,
                              const std::int32_t num_threads = 1) {
   
   if (vector_1.size() != vector_2.size()) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "BraketVector types do not match each other" << std::endl;
      ss << "dim_1 = " << vector_1.size() << ", dim_2 = " << vector_2.size() << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   vector_out->resize(vector_1.size());
   
#pragma omp parallel for num_threads(num_threads)
   for (std::int64_t i = 0; i < vector_1.size(); ++i) {
      (*vector_out)[i] = coeff_1*vector_1[i] + coeff_2*vector_2[i];
   }
   
}


template<typename T>
void CopyVector(std::vector<T> *copied,
                const std::vector<T> &v,
                const std::int32_t num_threads = 1) {
   
   copied->resize(v.size());
#pragma omp parallel for num_threads(num_threads)
   for (std::size_t i = 0; i < v.size(); ++i) {
      (*copied)[i] = v[i];
   }
}

//! @brief Calculate L1 distance, aka Manhattan distance,
//! \f$ \|c_1\boldsymbol{v}_1 - c_2\boldsymbol{v}_2\|_1=\sum_{i}\|c_1v_{1,i} - c_2v_{2,i}\| \f$.
//! @tparam T1 Value type of coeff_1.
//! @tparam T2 Value type of braket_vector_1.
//! @tparam T3 Value type of coeff_2.
//! @tparam T4 Value type of braket_vector_2.
//! @param coeff_1 The coefficient \f$ c_1 \f$.
//! @param vector_1 Vector \f$ \boldsymbol{v}_1\f$.
//! @param coeff_2 The coefficient \f$ c_2 \f$.
//! @param vector_2 Vector \f$ \boldsymbol{v}_2\f$.
//! @return L1 distance \f$ \|c_1\boldsymbol{v}_1 - c_2\boldsymbol{v}_2\|_1 \f$.
template<typename T1, typename T2, typename T3, typename T4>
auto CalculateL1Distance(const T1 coeff_1,
                         const std::vector<T2> &vector_1,
                         const T3 coeff_2,
                         const std::vector<T4> &vector_2,
                         const std::int32_t num_threads = 1)
-> decltype(std::declval<T1>()*std::declval<T2>() - std::declval<T3>()*std::declval<T4>()) {
   if (vector_1.size() != vector_2.size()) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "BraketVector types do not match each other" << std::endl;
      ss << "dim_1 = " << vector_1.size() << ", dim_2 = " << vector_2.size() << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   using T1T2T3T4 = decltype(std::declval<T1>()*std::declval<T2>() - std::declval<T3>()*std::declval<T4>());
   T1T2T3T4 val_out = 0;
      
#pragma omp parallel for reduction(+: val_out) num_threads(num_threads)
   for (std::size_t i = 0; i < vector_1.size(); ++i) {
      val_out += std::abs(coeff_1*vector_1[i] - coeff_2*vector_2[i]);
   }
   
   return val_out;
}


}
}


#endif /* COMPNAL_BLAS_VECTOR_VECTOR_OPERATION_HPP_ */
