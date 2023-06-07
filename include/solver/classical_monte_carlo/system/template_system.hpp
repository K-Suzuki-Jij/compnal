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

#include "../../../model/utility/variable.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

template<class ModelType, class RandType>
class System;

//! @brief Generate random spin configurations.
//! @tparam ModelType Model class.
//! @tparam RandType Random number engine class.
//! @param model The model.
//! @param random_number_engine Random number engine.
//! @return Random spin configurations.
template<class ModelType, class RandType>
std::vector<model::utility::Spin> GenerateRandomSpins(const ModelType &model,
                                                      RandType *random_number_engine) {
   
   const std::int32_t system_size = model.GetLattice().GetSystemSize();
   const std::vector<std::int32_t> &twice_spin_magnitude = model.GetTwiceSpinMagnitude();
   
   std::vector<model::utility::Spin> spins;
   spins.reserve(system_size);
   
   for (std::int32_t i = 0; i < system_size; ++i) {
      auto spin = model::utility::Spin{0.5*twice_spin_magnitude[i], model.GetSpinScaleFactor()};
      spin.SetStateRandomly(random_number_engine);
      spins.push_back(spin);
   }
   
   return spins;
}


} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_ */
