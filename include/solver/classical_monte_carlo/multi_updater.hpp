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
//  multi_updater.hpp
//  compnal
//
//  Created by kohei on 2023/10/30.
//  
//

#pragma once

#include "single_updater.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Run classical monte carlo simulation using multi flip.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @param system Pointer to the system.
//! @param num_update_variables The number of variables to be updated at once.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param seed Seed of random number engine.
//! @param update_method State update method.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType>
void MultiUpdater(SystemType *system,
                  const std::int32_t num_update_variables,
                  const std::int32_t num_sweeps,
                  const double beta,
                  const typename RandType::result_type seed,
                  const StateUpdateMethod update_method,
                  const SpinSelectionMethod spin_selector) {
   
   // If num_update_variables == 1, use SingleUpdater.
   if (num_update_variables == 1) {
      SingleUpdater<SystemType, RandType>(system, num_sweeps, beta, seed, update_method, spin_selector);
      return;
   }
   
   if (spin_selector != SpinSelectionMethod::RANDOM) {
      throw std::invalid_argument("Spin selection method must be random.");
   }
   
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

   if (num_update_variables == 2) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      std::uniform_int_distribution<std::int32_t> dist_system_size_m1(0, system_size - 2);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            // Choose two different indices randomly.
            const std::int32_t index_1 = dist_system_size(random_number_engine);
            std::int32_t index_2 = dist_system_size_m1(random_number_engine);
            if (index_1 <= index_2) {
               index_2++;
            }

            // Choose two different states randomly.
            const std::int32_t present_state_1 = system->GetSample()[index_1].GetStateNumber();
            const std::int32_t present_state_2 = system->GetSample()[index_2].GetStateNumber();
            const std::int32_t num_state_1 = system->GetSample()[index_1].GetNumState();
            const std::int32_t num_state_2 = system->GetSample()[index_2].GetNumState();
            std::int32_t new_whole_state = random_number_engine()%(num_state_1*num_state_2 - 1);
            if (0 <= new_whole_state && new_whole_state < num_state_1 - 1) {
               const std::int32_t candidate_state = system->GetSample()[index_1].GenerateCandidateState(&random_number_engine);
               if (trans_prob(beta*system->GetEnergyDifference(index_1, candidate_state), dist_real(random_number_engine))) {
                  system->Flip(index_1, candidate_state);
               }
            }
            else if (num_state_1 - 1 <= new_whole_state && new_whole_state < num_state_1 + num_state_2 - 2) {
               new_whole_state -= (num_state_1 - 1);
               const std::int32_t candidate_state = system->GetSample()[index_2].GenerateCandidateState(&random_number_engine);
               if (trans_prob(beta*system->GetEnergyDifference(index_2, candidate_state), dist_real(random_number_engine))) {
                  system->Flip(index_2, candidate_state);
               }
            }
            else {
               new_whole_state -= (num_state_1 - 1) + (num_state_2 - 1);
               std::int32_t new_state_1 = system->GetSample()[index_1].GenerateCandidateState(&random_number_engine);
               std::int32_t new_state_2 = system->GetSample()[index_2].GenerateCandidateState(&random_number_engine);
//               if (present_state_1 <= new_state_1) {
//                  new_state_1++;
//               }
//               if (present_state_2 <= new_state_2) {
//                  new_state_2++;
//               }
               if (trans_prob(beta*system->GetEnergyDifferenceTwoFlip(index_1, new_state_1, index_2, new_state_2), dist_real(random_number_engine))) {
                  system->Flip(index_1, new_state_1);
                  system->Flip(index_2, new_state_2);
               }
            }
         }
      }
   }
   else {
      throw std::invalid_argument("The number of variables to be updated at once must be 1, 2");
   }
   
}

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
