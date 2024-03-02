//
//  Copyright 2024 Kohei Suzuki
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
//  single_spin_flip.hpp
//  compnal
//
//  Created by Kohei Suzuki on 2024/03/02.
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Run classical monte carlo simulation using Metropolis method.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @tparam Updater Spin updater.
//! @param system Pointer to the system.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType, template<class, typename> class Updater>
void SingleSpinFlip(SystemType *system,
                    const std::int32_t num_sweeps,
                    const double beta,
                    const SpinSelectionMethod spin_selector) {
   
   const std::int32_t system_size = system->GetSystemSize();
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      auto updater = Updater<SystemType, RandType>(system, beta);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(system->GetRandomNumberEngine());
            system->Flip(index, updater.GetNewState(index));
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      auto updater = Updater<SystemType, RandType>(system, beta);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            system->Flip(i, updater.GetNewState(i));
         }
      }
   }
   else {
      throw std::invalid_argument("Unknown SpinSelectionMethod");
   }
   
}


} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
