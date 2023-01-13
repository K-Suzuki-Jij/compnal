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
//  blas_algorithm.hpp
//  compnal
//
//  Created by kohei on 2023/01/02.
//  
//

#ifndef COMPNAL_BLAS_BLAS_ALGORITHM_HPP_
#define COMPNAL_BLAS_BLAS_ALGORITHM_HPP_

namespace compnal {
namespace blas {

enum DiagAlgorithm {
  
   LANCZOS = 0,
   
   LOBPCG = 1,
   
   DAVIDSON = 2
   
};

enum LinearEqAlgorithm {
   
   CONJUGATE_GRADIENT = 0,
   
   MINIMUM_RESIDUAL = 1
   
};

} // namespace blas
} // namespace compnal


#endif /* COMPNAL_BLAS_BLAS_ALGORITHM_HPP_ */
