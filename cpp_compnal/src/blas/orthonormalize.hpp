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
//  orthonormalize.hpp
//  compnal
//
//  Created by kohei on 2023/01/15.
//  
//

#ifndef COMPNAL_BLAS_ORTHONORMALIZE_HPP_
#define COMPNAL_BLAS_ORTHONORMALIZE_HPP_

#include "vector_vector_operation.hpp"
#include <iostream>
#include <sstream>

namespace compnal {
namespace blas {


template<typename RealType>
void Orthonormalize(std::vector<RealType> *target_vector,
                    const std::vector<std::vector<RealType>> &vectors,
                    const std::int32_t num_threads = 1,
                    const bool flag_normalize = true) {
         
   for (std::size_t i = 0; i < vectors.size(); ++i) {
      if (target_vector->size() != vectors[i].size()) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Dimenstion of BraketVector list doen not match each other" << std::endl;
         ss << "Failed to orthonormalize." << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const RealType inner_product = CalculateInnerProduct(*target_vector, vectors[i], num_threads);
      
#pragma omp parallel for num_threads(num_threads)
      for (std::size_t j = 0; j < target_vector->size(); ++j) {
         (*target_vector)[j] -= inner_product*vectors[i][j];
      }
   }
   
   if (flag_normalize) {
      Normalize(target_vector, num_threads);
   }

}



}  // namespace blas
}  // namespace compnal


#endif /* COMPNAL_BLAS_ORTHONORMALIZE_HPP_ */
