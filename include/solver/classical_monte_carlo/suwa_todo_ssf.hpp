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

   const auto suwa_todo =[](std::vector<double> &w, std::int32_t now_state, std::vector<double> &work) {
      const std::int32_t num_state = static_cast<std::int32_t>(w.size());
      const std::int32_t max_ind = std::distance(w.begin(), std::max_element(w.begin(), w.end()));

      std::swap(w[0], w[max_ind]);

      if (now_state == max_ind) {
         now_state = 0;
      } 
      else if (now_state == 0) {
         now_state = max_ind;
      }

      work[1] = w[0];
      for (int i = 1; i < num_state; ++i) {
         work[i + 1] = work[i] + w[i];
      }
      work[0] = work[num_state];

      std::vector<double> prob(num_state);

      double prob_sum = 0.0;
      for (int j = 0; j < num_state; ++j) {
         double d_ij = work[now_state + 1] - work[j] + w[0];
         double a = std::min({d_ij, w[now_state] + w[j] - d_ij, w[now_state], w[j]});
         if (j == 0) {
            prob[max_ind] = std::max(0.0, a);
            prob_sum += prob[max_ind];
         }
         else if (j == max_ind) {
            prob[0] = std::max(0.0, a);
            prob_sum += prob[0];
         }
         else {
            prob[j] = std::max(0.0, a);
            prob_sum += prob[j];
         }
      }

      for (int j = 0; j < num_state; ++j) {
         prob[j] /= prob_sum;
      }

      return prob;
   };
   
   const auto get_new_state = [](const std::vector<double> &prob_list,
                                 const double dist_real) {
      const std::int32_t num_state = static_cast<std::int32_t>(prob_list.size());
      double prob_sum = 0.0;
      for (std::int32_t state = 0; state < num_state; state++) {
         if (dist_real < prob_list[state] + prob_sum) {
            return state;
         }
         prob_sum += prob_list[state];
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
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(random_number_engine);
            const std::int32_t num_state = system->GetNumState(index);
            for (std::int32_t state = 0; state < num_state; ++state) {
               dW[state] = std::exp(-beta*system->GetEnergyDifference(index, state));
            }    
            const auto &prob_list = suwa_todo(dW, system->GetStateNumber(index), S);
            std::int32_t new_state = get_new_state(prob_list, dist_real(random_number_engine));
            system->Flip(index, new_state);
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t num_state = system->GetNumState(i);
            for (std::int32_t state = 0; state < num_state; ++state) {
               dW[state] = std::exp(-beta*system->GetEnergyDifference(i, state));
            }    
            const auto &prob_list = suwa_todo(dW, system->GetStateNumber(i), S);
            std::int32_t new_state = get_new_state(prob_list, dist_real(random_number_engine));
            system->Flip(i, new_state);
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
