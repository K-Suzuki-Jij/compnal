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
//  parallel_tempering.hpp
//  compnal
//
//  Created by kohei on 2023/07/04.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_PARALLEL_TEMPERING_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_PARALLEL_TEMPERING_HPP_

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

template<class SystemType, typename RandType>
void ParallelTempering(std::vector<SystemType*> *system_list_pointer,
                       const std::int32_t num_sweeps,
                       const std::int32_t num_swaps,
                       const std::int32_t num_threads,
                       const typename RandType::result_type seed,
                       const std::vector<double> &beta_list,
                       const StateUpdateMethod updater,
                       const SpinSelectionMethod spin_selector) {
   
   if (system_list_pointer->size() != beta_list.size()) {
      throw std::invalid_argument("The size of system_list is not equal to beta_list.");
   }
   
   // Set random number engine
   RandType random_number_engine(seed);
   std::uniform_real_distribution<double> dist_real(0, 1);
   
   // Set update function
   std::function<bool(double, double, double)> trans_prob;
   if (updater == StateUpdateMethod::METROPOLIS) {
      trans_prob = [](const double delta_energy, const double beta, const double dist_real) {
         return delta_energy <= 0.0 || std::exp(-beta*delta_energy) > dist_real;
      };
   }
   else if (updater == StateUpdateMethod::HEAT_BATH) {
      trans_prob = [](const double delta_energy, const double beta, const double dist_real) {
         return 1.0/(1.0 + std::exp(beta*delta_energy)) > dist_real;
      };
   }
   else {
      throw std::invalid_argument("Unknown UpdateMethod");
   }
   
   
   const std::int32_t num_total = num_sweeps + num_swaps;
   std::int32_t sweep_count = num_sweeps;
   std::int32_t swap_count = num_swaps;
   
   if (spin_selector == SpinSelectionMethod::RANDOM) {
      for (std::int32_t i = 0; i < num_total; ++i) {
         const double a = sweep_count - (sweep_count + swap_count)*num_sweeps/num_total;
         const double b = swap_count - (sweep_count + swap_count)*num_swaps/num_total;
         if (a >= b) {
            // Do Sweep
            for (std::size_t j = 0; j < beta_list.size(); ++j) {
               const std::int32_t system_size = (*(*system_list_pointer)[j]).GetSystemSize();
               std::uniform_int_distribution<std::int32_t> dist_system_size(0, system_size - 1);
               for (std::int32_t k = 0; k < system_size; k++) {
                  const std::int32_t index = dist_system_size(random_number_engine);
                  const auto candidate_state = (*(*system_list_pointer)[j]).GenerateCandidateState(index);
                  const auto delta_energy = (*(*system_list_pointer)[j]).GetEnergyDifference(index, candidate_state);
                  if (trans_prob(delta_energy, beta_list[j], dist_real(random_number_engine))) {
                     (*(*system_list_pointer)[j]).Flip(index, candidate_state);
                  }
               }
            }
         }
         else {
            // Do Replica Swap
            // Even Number Replica
            const std::int32_t num_even_swap = static_cast<std::int32_t>(beta_list.size()/2) + beta_list.size()%2;
            const std::int32_t num_odd_swap = static_cast<std::int32_t>(beta_list.size()/2);
            for (std::int32_t j = 0; j < num_even_swap; ++j) {
               const std::int32_t ind = 2*j;
               const auto delta_energy = (*(*system_list_pointer)[ind + 1]).GetEnergy() - (*(*system_list_pointer)[ind]).GetEnergy();
               const auto delta_beta = beta_list[ind + 1] - beta_list[ind];
               if (trans_prob(delta_energy, delta_beta, dist_real(random_number_engine))) {
                  std::swap((*system_list_pointer)[ind + 1], (*system_list_pointer)[ind]);
               }
            }
            // Odd Number Replica
            for (std::int32_t j = 0; j < num_odd_swap; ++j) {
               const std::int32_t ind = 2*j + 1;
               const auto delta_energy = (*(*system_list_pointer)[ind + 1]).GetEnergy() - (*(*system_list_pointer)[ind]).GetEnergy();
               const auto delta_beta = beta_list[ind + 1] - beta_list[ind];
               if (trans_prob(delta_energy, delta_beta, dist_real(random_number_engine))) {
                  std::swap((*system_list_pointer)[ind + 1], (*system_list_pointer)[ind]);
               }
            }
         }
      }
   }
   else if (spin_selector == SpinSelectionMethod::SEQUENTIAL) {
      
   }
   else {
      throw std::invalid_argument("Unknown SpinSelectionMethod");
   }
   
}

std::vector<double> GenerateBetaList(const std::pair<double, double> temperature_range,
                                     const std::int32_t num_replica,
                                     const TemperatureDistribution temperature_distribution) {
   
   std::vector<double> beta_list(num_replica);
   const double beta_min = 1.0/temperature_range.second;
   const double beta_max = 1.0/temperature_range.first;

   if (num_replica == 1) {
      return std::vector<double>{beta_min};
   }
   
   if (temperature_distribution == TemperatureDistribution::LINEAR) {
      for (std::int32_t i = 0; i < num_replica; ++i) {
         beta_list[i] = beta_min + i*(beta_max - beta_min)/(num_replica - 1);
      }
      return beta_list;
   }
   else if (temperature_distribution == TemperatureDistribution::GEOMETRIC) {
      const double alpha = std::pow(beta_max/beta_min, 1/static_cast<double>(num_replica - 1));
      double beta = beta_min;
      for (std::int32_t i = 0; i < num_replica; ++i) {
         beta_list[i] = beta;
         beta = beta*alpha;
      }
   }
   else {
      throw std::runtime_error("Unknwon beta schedule list");
   }
   
   return beta_list;
}



} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_PARALLEL_TEMPERING_HPP_ */
