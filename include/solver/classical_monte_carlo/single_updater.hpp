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
//  single_updater.hpp
//  compnal
//
//  Created by kohei on 2023/06/14.
//  
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Run classical monte carlo simulation using single flip.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @param system Pointer to the system.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param seed Seed of random number engine.
//! @param update_method State update method.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType>
void SingleUpdater(SystemType *system,
                   const std::int32_t num_sweeps,
                   const double beta,
                   const typename RandType::result_type seed,
                   const StateUpdateMethod update_method,
                   const SpinSelectionMethod spin_selector) {
   
   const std::int32_t system_size = system->GetSystemSize();
   
   // Set random number engine
   RandType random_number_engine(seed);
   std::uniform_real_distribution<double> dist_real(0, 1);
   
   // Set update function
   std::function<bool(double, double)> trans_prob;
   if (update_method == StateUpdateMethod::METROPOLIS) {
      trans_prob = [](const double delta_S, const double dist_real) {
         return delta_S <= 0.0 || std::exp(-delta_S) > dist_real;
      };
   }
   else if (update_method == StateUpdateMethod::HEAT_BATH) {
      trans_prob = [](const double delta_S, const double dist_real) {
         return 1.0/(1.0 + std::exp(delta_S)) > dist_real;
      };
   }
   else {
      throw std::invalid_argument("Unknown UpdateMethod");
   }
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(random_number_engine);
            const auto candidate_state = system->GenerateCandidateState(index);
            const auto delta_energy = system->GetEnergyDifference(index, candidate_state);
            if (trans_prob(delta_energy*beta, dist_real(random_number_engine))) {
               system->Flip(index, candidate_state);
            }
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const auto candidate_state = system->GenerateCandidateState(i);
            const auto delta_energy = system->GetEnergyDifference(i, candidate_state);
            if (trans_prob(delta_energy*beta, dist_real(random_number_engine))) {
               system->Flip(i, candidate_state);
            }
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
