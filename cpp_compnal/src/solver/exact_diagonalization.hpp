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
//  exact_diagonalization.hpp
//  compnal
//
//  Created by kohei on 2023/01/02.
//  
//

#ifndef COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_
#define COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_

#include "../blas/all.hpp"

#include <vector>

namespace compnal {
namespace solver {

template<class ModelType>
class ExactDiag {
   using RealType = typename ModelType::ValueType;
   using CQNType = typename ModelType::CQNType;
   using CQNHash = typename ModelType::CQNHash;
   
public:
   ExactDiag(const ModelType &model): model_(model) {}
   
   void Diagonaliza() {
      
   }
   
   
   
   
private:
   std::vector<std::pair<RealType, std::vector<RealType>>> eigenvalues_;
   const ModelType model_;
   
   std::unordered_map<CQNType, std::vector<std::int64_t>, CQNHash> bases_;
   std::unordered_map<CQNType, std::unordered_map<std::int64_t, std::int64_t>, CQNHash> bases_inv_;

   
   //! @brief Diagonalization method.
   blas::DiagAlgorithm diag_method_ = blas::DiagAlgorithm::LANCZOS;
   
   
   
   
};


} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_ */
