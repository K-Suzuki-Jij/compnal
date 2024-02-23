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
//  heat_bath_updater.hpp
//  compnal
//
//  Created by 鈴木浩平 on 2024/02/23.
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Run classical monte carlo simulation using heat bath method.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @param system Pointer to the system.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param seed Seed of random number engine.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType>
void HeatBathUpdater(SystemType *system,
                     const std::int32_t num_sweeps,
                     const double beta,
                     const typename RandType::result_type seed,
                     const SpinSelectionMethod spin_selector) {
   
   const std::int32_t system_size = system->GetSystemSize();
   
   // Set random number engine
   RandType random_number_engine(seed);
   std::uniform_real_distribution<double> dist_real(0, 1);
   
   //Find max spin
   std::int32_t max_num_state = 0;
   for (std::int32_t i = 0; i < system_size; i++) {
      if (max_num_state < system->GetNumState(i)) {
         max_num_state = system->GetNumState(i);
      }
   }
   
   std::vector<double> prob_list(max_num_state);
   
   const auto trans_prob = [](const std::vector<double> &prob_list,
                              const std::int32_t num_stete,
                              const double norm,
                              const double dist_real) {
      double prob_sum = 0.0;
      for (std::int32_t state = 0; state < num_stete; state++) {
         if (dist_real < prob_list[state]/norm) {
            return state;
         }
         prob_sum += prob_list[state]/norm;
      }
      return num_stete - 1;
   };
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(random_number_engine);
            const std::int32_t num_state = system->GetNumState(index);
            double z = 0.0;
            for (std::int32_t state = 0; state < num_state; ++state) {
               prob_list[state] = std::exp(-system->GetEnergyDifference(index, state));
               z += prob_list[state];
            }
            system->Flip(index, trans_prob(prob_list, num_state, z, dist_real(random_number_engine)));
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t num_state = system->GetNumState(i);
            double z = 0.0;
            for (std::int32_t state = 0; state < num_state; ++state) {
               prob_list[state] = std::exp(-system->GetEnergyDifference(i, state));
               z += prob_list[state];
            }
            system->Flip(i, trans_prob(prob_list, num_state, z, dist_real(random_number_engine)));
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
