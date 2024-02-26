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
//  metropolis_pt.hpp
//  compnal
//
//  Created by kohei on 2023/07/04.
//
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Running classical monte carlo simulation using parallel tempering.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @param system_list_pointer Pointer to the list of system.
//! @param num_sweeps The number of sweeps.
//! @param num_swaps The number of swaps for each replica.
//! @param seed Seed of random number engine.
//! @param beta_list List of inverse temperature.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType>
void MetropolisPT(std::vector<SystemType*> *system_list_pointer,
                  const std::int32_t num_sweeps,
                  const std::int32_t num_swaps,
                  const typename RandType::result_type seed,
                  const std::vector<double> &beta_list,
                  const SpinSelectionMethod spin_selector) {
   
   if (system_list_pointer->size() != beta_list.size()) {
      throw std::invalid_argument("The size of system_list is not equal to beta_list.");
   }
   
   // Set random number engine
   RandType random_number_engine(seed);
   std::uniform_real_distribution<double> dist_real(0, 1);
   
   // Set update function
   const auto trans_prob = [](const double delta_S, const double dist_real) {
      return delta_S <= 0.0 || std::exp(-delta_S) > dist_real;
   };
   
   const std::int32_t num_total = num_sweeps + num_swaps;
   std::int32_t sweep_count = num_sweeps;
   std::int32_t swap_count = num_swaps;
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      for (std::int32_t i = 0; i < num_total; ++i) {
         const double a = sweep_count - static_cast<double>(sweep_count + swap_count)*num_sweeps/num_total;
         const double b = swap_count - static_cast<double>(sweep_count + swap_count)*num_swaps/num_total;
         if (a >= b) {
            // Do Sweep
            for (std::size_t j = 0; j < beta_list.size(); ++j) {
               const std::int32_t system_size = (*(*system_list_pointer)[j]).GetSystemSize();
               std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
               for (std::int32_t k = 0; k < system_size; k++) {
                  const std::int32_t index = dist_system_size(random_number_engine);
                  const auto candidate_state = (*(*system_list_pointer)[j]).GenerateCandidateState(index);
                  const auto delta_energy = (*(*system_list_pointer)[j]).GetEnergyDifference(index, candidate_state);
                  if (trans_prob(delta_energy*beta_list[j], dist_real(random_number_engine))) {
                     (*(*system_list_pointer)[j]).Flip(index, candidate_state);
                  }
               }
            }
            sweep_count--;
         }
         else {
            // Do Replica Swap
            const std::int32_t num_swap = static_cast<std::int32_t>(beta_list.size()) - 1;
            for (std::int32_t ind = 0; ind < num_swap; ++ind) {
               const auto delta_energy = (*(*system_list_pointer)[ind + 1]).GetEnergy() - (*(*system_list_pointer)[ind]).GetEnergy();
               const auto delta_beta = beta_list[ind + 1] - beta_list[ind];
               if (trans_prob(-delta_energy*delta_beta, dist_real(random_number_engine))) {
                  std::swap((*system_list_pointer)[ind], (*system_list_pointer)[ind + 1]);
               }
            }
            swap_count--;
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      for (std::int32_t i = 0; i < num_total; ++i) {
         const double a = sweep_count - static_cast<double>(sweep_count + swap_count)*num_sweeps/num_total;
         const double b = swap_count - static_cast<double>(sweep_count + swap_count)*num_swaps/num_total;
         if (a >= b) {
            // Do Sweep
            for (std::size_t j = 0; j < beta_list.size(); ++j) {
               const std::int32_t system_size = (*(*system_list_pointer)[j]).GetSystemSize();
               for (std::int32_t index = 0; index < system_size; index++) {
                  const auto candidate_state = (*(*system_list_pointer)[j]).GenerateCandidateState(index);
                  const auto delta_energy = (*(*system_list_pointer)[j]).GetEnergyDifference(index, candidate_state);
                  if (trans_prob(delta_energy*beta_list[j], dist_real(random_number_engine))) {
                     (*(*system_list_pointer)[j]).Flip(index, candidate_state);
                  }
               }
            }
            sweep_count--;
         }
         else {
            // Do Replica Swap
            const std::int32_t num_swap = static_cast<std::int32_t>(beta_list.size()) - 1;
            for (std::int32_t ind = 0; ind < num_swap; ++ind) {
               const auto delta_energy = (*(*system_list_pointer)[ind + 1]).GetEnergy() - (*(*system_list_pointer)[ind]).GetEnergy();
               const auto delta_beta = beta_list[ind + 1] - beta_list[ind];
               if (trans_prob(-delta_energy*delta_beta, dist_real(random_number_engine))) {
                  std::swap((*system_list_pointer)[ind], (*system_list_pointer)[ind + 1]);
               }
            }
            swap_count--;
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
