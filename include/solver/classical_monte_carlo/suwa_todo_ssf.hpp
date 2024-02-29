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
   
   const auto get_new_state = [](const std::vector<double> &prob_list,
                                 const double z,
                                 const std::int32_t num_state,
                                 const double dist_real) {
      double prob_sum = 0.0;
      for (std::int32_t state = 0; state < num_state; state++) {
         if (dist_real < prob_list[state]/z + prob_sum) {
            return state;
         }
         prob_sum += prob_list[state]/z;
      }
      return num_state - 1;
   };

   // Find Max number of state
   std::int32_t max_num_state = 0;
   for (std::int32_t i = 0; i < system_size; i++) {
      max_num_state = std::max(max_num_state, system->GetNumState(i));
   }
   
   std::vector<double> dW(max_num_state);
   std::vector<double> S(max_num_state + 1);
   std::vector<double> prob_list(max_num_state);
   std::vector<std::int32_t> indices(max_num_state);
   std::iota(indices.begin(), indices.end(), 0);

   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
//            const std::int32_t index = dist_system_size(random_number_engine);
//            const std::int32_t num_state = system->GetNumState(index);
//            for (std::int32_t state = 0; state < num_state; ++state) {
//               dW[state] = std::exp(-beta*system->GetEnergyDifference(index, state));
//            }    
//            const auto &prob_list = suwa_todo(dW, system->GetStateNumber(index), S);
//            std::int32_t new_state = get_new_state(prob_list, dist_real(random_number_engine));
//            system->Flip(index, new_state);
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t num_state = system->GetNumState(i);
            std::int32_t max_ind = 0;
            for (std::int32_t state = 0; state < num_state; ++state) {
               dW[state] = std::exp(-beta*system->GetEnergyDifference(i, state));
               if (dW[max_ind] < dW[state]) {
                  max_ind = state;
               }
            }
            const std::int32_t now_state = system->GetStateNumber(i);
            indices[0] = max_ind;
            indices[max_ind] = 0;
            
            S[1] = dW[0];
            for (std::int32_t i = 1; i < num_state; ++i) {
               S[i + 1] = S[i] + dW[indices[i]];
            }
            S[0] = S[num_state];
            
            double prob_sum = 0.0;
            for (std::int32_t j = 0; j < num_state; ++j) {
               const double d_ij = S[indices[now_state] + 1] - S[j] + dW[indices[0]];
               const double a = std::min({d_ij, dW[now_state] + dW[indices[j]] - d_ij, dW[now_state], dW[indices[j]]});
               prob_list[indices[j]] = std::max(0.0, a);
               prob_sum += prob_list[indices[j]];
            }
            
            system->Flip(i, get_new_state(prob_list, prob_sum, num_state, dist_real(random_number_engine)));
            indices[0] = 0;
            indices[max_ind] = max_ind;

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
