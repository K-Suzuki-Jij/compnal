//
//  Copyright 2022 Kohei Suzuki
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
//  ssf_updater.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_SSF_UPDATER_HPP_
#define COMPNAL_SOLVER_CMC_UTILITY_SSF_UPDATER_HPP_

#include <random>

namespace compnal {
namespace solver {
namespace cmc_utility {

template<class SystemType>
void SSFUpdater(SystemType *system,
                const std::int32_t num_sweeps,
                const typename SystemType::ValueType beta,
                const std::uint64_t seed,
                const std::function<bool(typename SystemType::ValueType, typename SystemType::ValueType, typename SystemType::ValueType)> ssf_transition_probability) {
   
   const std::int32_t system_size = system->GetSystemSize();
   
   // Set random number engine
   utility::RandType random_number_engine(seed);
   std::uniform_real_distribution<typename SystemType::ValueType> dist_real(0, 1);
   std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
   
   // Do Metropolis update
   for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
      for (std::int32_t i = 0; i < system_size; i++) {
         const std::int32_t index = dist_system_size(random_number_engine);
         if (ssf_transition_probability(system->GetEnergyDifference(index), beta, dist_real(random_number_engine))) {
            system->Flip(index);
         }
      }
   }
   
}

/*
template<class SystemType>
void IRKMRUpdater(SystemType *system,
                     const std::int32_t num_sweeps,
                     const typename SystemType::ValueType beta,
                     const std::uint64_t seed) {
   
   using RealType = typename SystemType::ValueType;
   const std::int32_t system_size = system->GetSystemSize();
   
   std::vector<RealType> w(system_size + 1);
   std::vector<RealType> s(system_size + 1);
   std::vector<RealType> flip_index(system_size + 1);
   std::int32_t current_index = system_size;
   
   
   //Set weight
   for (std::int32_t i = 0; i < system_size; ++i) {
      w[i] = std::exp(-beta*system->GetEnergyDifference(i));
      flip_index[i] = i;
   }
   w[system_size] = 1.0;
   flip_index[system_size] = system_size;
   
   //Find max weight
   const auto first_max_w_index = std::distance(w.begin(), std::max_element(w.begin(), w.end()));
   std::swap(w[0], w[first_max_w_index]);
   std::swap(flip_index[0], flip_index[first_max_w_index]);
   if (first_max_w_index == system_size) {
      current_index = 0;
   }

   //Set weight sum
   s[0] = w[0];
   for (std::int32_t i = 0; i < system_size; ++i) {
      s[i + 1] = s[i] + w[i + 1];
   }
   
   // Set random number engine
   utility::RandType random_number_engine(seed);
   std::uniform_real_distribution<RealType> dist_real(0, 1);
   std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
   
   // Do Metropolis update
   for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
      for (std::int32_t i = 0; i < system_size; i++) {
         std::int32_t index = dist_system_size(random_number_engine);
         if (index >= current_index) {
            index++;
         }
         RealType a = 0.0;
         if (index == 0) {
            a = s[current_index] - s[system_size] + w[0];
         }
         else {
            a = s[current_index] - s[index - 1] + w[0];
         }
         
         const RealType b = w[current_index] + w[index] - a;
         const RealType c = w[current_index];
         const RealType d = w[index];
         if (dist_real(random_number_engine) <= std::min({a, b, c, d})) {
            system->Flip(flip_index[index]);
            
            //
            //Set weight
            for (std::int32_t j = 0; j < system_size; ++j) {
               w[j] = std::exp(-beta*system->GetEnergyDifference(j));
               flip_index[j] = j;
            }
            w[system_size] = 1.0;
            flip_index[system_size] = system_size;
            current_index = system_size;
            
            //Find max weight
            const auto max_w_index = std::distance(w.begin(), std::max_element(w.begin(), w.end()));
            std::swap(w[0], w[max_w_index]);
            std::swap(flip_index[0], flip_index[max_w_index]);
            if (max_w_index == system_size) {
               current_index = 0;
            }
            
            //Set weight sum
            s[0] = w[0];
            for (std::int32_t i = 0; i < system_size; ++i) {
               s[i + 1] = s[i] + w[i + 1];
            }
            //
            
         }
      }
   }
}
 */




} // namespace cmc_utility
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CMC_UTILITY_SSF_UPDATER_HPP_ */
