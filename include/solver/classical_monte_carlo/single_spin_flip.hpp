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
//  single_spin_flip.hpp
//  compnal
//
//  Created by Kohei Suzuki on 2024/03/02.
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Run classical monte carlo simulation using Metropolis method.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @tparam Updater Spin updater.
//! @param system Pointer to the system.
//! @param num_sweeps The number of sweeps.
//! @param beta Inverse temperature.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType, class Updater>
void SingleSpinFlip(SystemType *system,
                    const std::int32_t num_sweeps,
                    const double beta,
                    const SpinSelectionMethod spin_selector) {
   
   const std::int32_t system_size = system->GetSystemSize();
   auto updater = Updater(system->GetMaxNumState());
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            const std::int32_t index = dist_system_size(system->GetRandomNumberEngine());
            system->Flip(index, updater.GetNewState(system, index, beta));
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      for (std::int32_t sweep_count = 0; sweep_count < num_sweeps; sweep_count++) {
         for (std::int32_t i = 0; i < system_size; i++) {
            system->Flip(i, updater.GetNewState(system, i, beta));
         }
      }
   }
   else {
      throw std::invalid_argument("Unknown SpinSelectionMethod");
   }
   
}

//! @brief Running classical monte carlo simulation using parallel tempering.
//! @tparam SystemType System class.
//! @tparam RandType Random number engine class.
//! @tparam Updater Spin updater.
//! @param system_list_pointer Pointer to the list of system.
//! @param num_sweeps The number of sweeps.
//! @param num_swaps The number of swaps for each replica.
//! @param beta_list List of inverse temperature.
//! @param spin_selector Spin selection method.
template<class SystemType, typename RandType, class Updater>
void ParallelTempering(std::vector<SystemType*> *system_list_pointer,
                       const std::int32_t num_sweeps,
                       const std::int32_t num_swaps,
                       const std::vector<double> &beta_list,
                       const SpinSelectionMethod spin_selector) {
   
   if (system_list_pointer->size() != beta_list.size()) {
      throw std::invalid_argument("The size of system_list is not equal to beta_list.");
   }
   
   const std::int32_t num_total = num_sweeps + num_swaps;
   std::int32_t sweep_count = num_sweeps;
   std::int32_t swap_count  = num_swaps;
   
   // All systems differ only in temperature and all other parameters are the same.
   auto updater = Updater((*system_list_pointer)[0]->GetMaxNumState());
   const std::int32_t system_size = (*system_list_pointer)[0]->GetSystemSize();
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
      for (std::int32_t i = 0; i < num_total; ++i) {
         const double a = sweep_count - static_cast<double>(sweep_count + swap_count)*num_sweeps/num_total;
         const double b = swap_count - static_cast<double>(sweep_count + swap_count)*num_swaps/num_total;
         if (a >= b) {
            // Do Sweep
            for (std::size_t j = 0; j < beta_list.size(); ++j) {
               for (std::int32_t k = 0; k < system_size; k++) {
                  const std::int32_t index = dist_system_size((*system_list_pointer)[j]->GetRandomNumberEngine());
                  (*system_list_pointer)[j]->Flip(index, updater.GetNewState((*system_list_pointer)[j], index, beta_list[j]));
               }
            }
            sweep_count--;
         }
         else {
            // Do Replica Swap
            const std::int32_t num_swap = static_cast<std::int32_t>(beta_list.size()) - 1;
            for (std::int32_t ind = 0; ind < num_swap; ++ind) {
               const double delta_energy = (*system_list_pointer)[ind + 1]->GetEnergy() - (*system_list_pointer)[ind]->GetEnergy();
               const double delta_beta = beta_list[ind + 1] - beta_list[ind];
               if (updater.DecideAcceptance(&(*system_list_pointer)[ind]->GetRandomNumberEngine(), delta_energy, delta_beta)) {
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
               for (std::int32_t index = 0; index < system_size; index++) {
                  (*system_list_pointer)[j]->Flip(index, updater.GetNewState((*system_list_pointer)[j], index, beta_list[j]));
               }
            }
            sweep_count--;
         }
         else {
            // Do Replica Swap
            const std::int32_t num_swap = static_cast<std::int32_t>(beta_list.size()) - 1;
            for (std::int32_t ind = 0; ind < num_swap; ++ind) {
               const double delta_energy = (*system_list_pointer)[ind + 1]->GetEnergy() - (*system_list_pointer)[ind]->GetEnergy();
               const double delta_beta = beta_list[ind + 1] - beta_list[ind];
               if (updater.DecideAcceptance(&(*system_list_pointer)[ind]->GetRandomNumberEngine(), delta_energy, delta_beta)) {
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
