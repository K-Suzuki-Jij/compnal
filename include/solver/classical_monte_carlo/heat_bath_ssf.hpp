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
//  heat_bath_ssf.hpp
//  compnal
//
//  Created by 鈴木浩平 on 2024/02/23.
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

template<class SystemType, typename RandType>
class HeatBathUpdater {
   
public:
   HeatBathUpdater(const SystemType &system,
                   const double beta,
                   const typename RandType::result_type seed):
   system_(system), beta_(beta), random_number_engine_(seed), dist_real_(0, 1) {}
   
   std::int32_t GetNewState(const std::int32_t index) {
      const std::int32_t num_state = system_.GetNumState(index);
      double z = 0.0;
      for (std::int32_t state = 0; state < num_state; ++state) {
         z += std::exp(-beta_*system_.GetEnergyDifference(index, state));
      }
      double prob_sum = 0.0;
      const double dist = dist_real_(random_number_engine_);
      for (std::int32_t state = 0; state < num_state; ++state) {
         prob_sum += std::exp(-beta_*system_.GetEnergyDifference(index, state))/z;
         if (dist < prob_sum) {
            return state;
         }
      }
      return num_state - 1;
   }
   
private:
   const SystemType &system_;
   const double beta_;
   RandType random_number_engine_;
   std::uniform_real_distribution<double> dist_real_;
   
};

//! @brief Run classical monte carlo simulation using heat bath method.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @param system Pointer to the system.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param seed Seed of random number engine.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType>
void HeatBathSSF(SystemType *system,
                 const std::int32_t num_sweeps,
                 const double beta,
                 const typename RandType::result_type seed,
                 const SpinSelectionMethod spin_selector) {
   
   const std::int32_t system_size = system->GetSystemSize();
      
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      RandType random_number_engine(seed);
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      auto updater = HeatBathUpdater<SystemType, RandType>(*system, beta, random_number_engine());
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(random_number_engine);
            system->Flip(index, updater.GetNewState(index));
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      auto updater = HeatBathUpdater<SystemType, RandType>(*system, beta, seed);
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
