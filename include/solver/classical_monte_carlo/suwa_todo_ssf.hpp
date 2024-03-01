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

template<class SystemType, typename RandType>
class SuwaTodoUpdater {
  
public:
   SuwaTodoUpdater(const std::int32_t max_num_state,
                   const SystemType &system,
                   const double beta,
                   const typename RandType::result_type seed):
   system_(system), beta_(beta), random_number_engine_(seed), dist_real(0, 1) {
      weight_list_.resize(max_num_state);
      sum_weight_list_.resize(max_num_state + 1);
   }
   
   std::int32_t GetNewState(const std::int32_t index) {
      return GetNewStateAny(index);
   }
   
private:
   const SystemType &system_;
   const double beta_;
   RandType random_number_engine_;
   std::uniform_real_distribution<double> dist_real;
   std::vector<double> weight_list_;
   std::vector<double> sum_weight_list_;
   
   std::int32_t GetNewStateAny(const std::int32_t index) {
      const std::int32_t num_state = system_.GetNumState(index);
      const std::int32_t max_weight_state = system_.GetMaxBoltzmannWeightStateNumber(index);
      
      weight_list_[0] = std::exp(-beta_*system_.GetEnergyDifference(index, max_weight_state));
      sum_weight_list_[1] = weight_list_[0];
      
      for (std::int32_t state = 1; state < num_state; ++state) {
         if (state == max_weight_state) {
            weight_list_[state] = std::exp(-beta_*system_.GetEnergyDifference(index, 0));
         }
         else {
            weight_list_[state] = std::exp(-beta_*system_.GetEnergyDifference(index, state));
         }
         sum_weight_list_[state + 1] = sum_weight_list_[state] + weight_list_[state];
      }
      sum_weight_list_[0] = sum_weight_list_[num_state];
      
      const std::int32_t temp = system_.GetStateNumber(index);
      const std::int32_t now_state = (temp == 0) ? max_weight_state : ((temp == max_weight_state) ? 0 : temp);
      double prob_sum = 0.0;
      const double dist = dist_real(random_number_engine_);
      for (std::int32_t j = 0; j < num_state; ++j) {
         const double d_ij = sum_weight_list_[now_state + 1] - sum_weight_list_[j] + sum_weight_list_[1];
         prob_sum += std::max(0.0, std::min({d_ij, 1.0 + weight_list_[j] - d_ij, 1.0, weight_list_[j]}));
         if (dist < prob_sum) {
            return (j == max_weight_state) ? 0 : ((j == 0) ? max_weight_state : j);
         }
      }
      return num_state - 1;
   }

};

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
   
   // Find Max number of state
   std::int32_t max_num_state = 0;
   for (std::int32_t i = 0; i < system_size; i++) {
      max_num_state = std::max(max_num_state, system->GetNumState(i));
   }
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      RandType random_number_engine(seed);
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      auto updater = SuwaTodoUpdater<SystemType, RandType>(max_num_state, *system, beta, random_number_engine());
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(random_number_engine);
            system->Flip(index, updater.GetNewState(index));
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      auto updater = SuwaTodoUpdater<SystemType, RandType>(max_num_state, *system, beta, seed);
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
