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
//  suwa_todo_ssf.hpp
//  compnal
//
//  Created by 鈴木浩平 on 2024/02/25.
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Run classical monte carlo simulation using Metropolis method.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @param system Pointer to the system.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param seed Seed of random number engine.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType>
void SuwaTodoSSF(SystemType *system,
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
   
   const auto get_new_state = [](const std::vector<double> &prob_list,
                                 const std::int32_t num_stete,
                                 const double dist_real) {
      double prob_sum = 0.0;
      for (std::int32_t state = 1; state <= num_stete; state++) {
         if (dist_real < prob_list[state] + prob_sum) {
            return state - 1;
         }
         prob_sum += prob_list[state];
      }
      return num_stete - 1;
   };
   
   std::vector<double> dW(max_num_state + 1);
   std::vector<double> S(max_num_state + 1);
   std::vector<std::int32_t> s_ind(max_num_state + 1);
   std::vector<double> prob_list(max_num_state + 1);
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(random_number_engine);
            const std::int32_t num_state = system->GetNumState(index);
            double max_weight_index = 1;
            // Set wieight
            for (std::int32_t state = 1; state <= num_state; ++state) {
               dW[state] = std::exp(-beta*system->GetEnergyDifference(index, state - 1));
               s_ind[state] = state;
               if (dW[max_weight_index] < dW[state]) {
                  max_weight_index = state;
               }
            }
            
            // Assume dW[1] is the maximum
            std::swap(s_ind[1], s_ind[max_weight_index]);
            
            // Set weight sum
            S[1] = dW[s_ind[1]];
            for (std::int32_t state = 1; state < num_state; ++state) {
               S[state + 1] = dW[s_ind[state + 1]] + S[state];
            }
            S[0] = S[num_state];
            
            // Set probability
            std::int32_t now_state = system->GetStateNumber(index);
            for (std::int32_t state = 1; state < num_state; ++state) {
               double d = S[now_state] - S[state - 1] + dW[s_ind[1]];
               double p2 = dW[s_ind[now_state]] + dW[s_ind[state]] - d;
               const double x = std::min({d, p2, dW[s_ind[now_state]], dW[s_ind[state]]});
               prob_list[state] = std::max({0.0, x});
            }
            system->Flip(index, get_new_state(prob_list, num_state, dist_real(random_number_engine)));
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      
      
   }
   else {
      throw std::invalid_argument("Unknown SpinSelectionMethod");
   }
   
   
}



} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
