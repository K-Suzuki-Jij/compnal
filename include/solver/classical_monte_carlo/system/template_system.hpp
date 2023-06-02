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
//  template_system.hpp
//  compnal
//
//  Created by kohei on 2023/05/06.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

template<class ModelType, class RandType>
class System;

template<class ModelType, class RandType>
std::vector<typename ModelType::PHQType> GenerateRandomSpins(const ModelType &model) {
   
}


} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_ */
