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
//  single_spin_flip_updater.hpp
//  compnal
//
//  Created by 鈴木浩平 on 2024/02/25.
//

#pragma once

namespace compnal {
namespace solver {
namespace classical_monte_carlo {


class MetropolisUpdater {
   
public:
   MetropolisUpdater(const std::int32_t max_num_state): max_num_state_(max_num_state), dist_real_(0, 1) {}

   template<class SystemType>
   std::int32_t GetNewState(SystemType *system, const std::int32_t index, const double beta) {
      const std::int32_t candidate_state = system->GenerateCandidateState(index);
      const double delta_energy = system->GetEnergyDifference(index, candidate_state);
      if (delta_energy <= 0.0 || std::exp(-beta*delta_energy) > dist_real_(system->GetRandomNumberEngine())) {
         return candidate_state;
      }
      else {
         return system->GetStateNumber(index);
      }
   }
   
   template<typename RandType>
   bool DecideAcceptance(RandType *random_number_engine, const double delta_energy, const double beta) {
      return delta_energy <= 0.0 || std::exp(-beta*delta_energy) > dist_real_(*random_number_engine);
   }
   
private:
   const std::int32_t max_num_state_;
   std::uniform_real_distribution<double> dist_real_;
   
};


class HeatBathUpdater {
   
public:
   HeatBathUpdater(const std::int32_t max_num_state): max_num_state_(max_num_state), dist_real_(0, 1) {
      prob_list.resize(max_num_state_);
   }
   
   template<class SystemType>
   std::int32_t GetNewState(SystemType *system, const std::int32_t index, const double beta) {
      const std::int32_t num_state = system->GetNumState(index);
      double z = 0.0;
      for (std::int32_t state = 0; state < num_state; ++state) {
         prob_list[state] = std::exp(-beta*system->GetEnergyDifference(index, state));
         z += prob_list[state];
      }
      z = 1.0/z;
      double prob_sum = 0.0;
      const double dist = dist_real_(system->GetRandomNumberEngine());
      for (std::int32_t state = 0; state < num_state; ++state) {
         prob_sum += z*prob_list[state];
         if (dist < prob_sum) {
            return state;
         }
      }
      return num_state - 1;
   }
   
   template<typename RandType>
   bool DecideAcceptance(RandType *random_number_engine, const double delta_energy, const double beta) {
      return 1.0/(1.0 + std::exp(-beta*delta_energy)) > dist_real_(*random_number_engine);
   }
   
private:
   const std::int32_t max_num_state_;
   std::uniform_real_distribution<double> dist_real_;
   std::vector<double> prob_list;
   
};


class SuwaTodoUpdater {
  
public:
   SuwaTodoUpdater(const std::int32_t max_num_state): max_num_state_(max_num_state), dist_real_(0, 1) {
      weight_list_.resize(max_num_state_);
      sum_weight_list_.resize(max_num_state_ + 1);
   }
   
   template<class SystemType>
   std::int32_t GetNewState(SystemType *system, const std::int32_t index, const double beta) {
      const std::int32_t num_state = system->GetNumState(index);
      const std::int32_t max_weight_state = system->GetMaxBoltzmannWeightStateNumber(index);
      
      weight_list_[0] = std::exp(-beta*system->GetEnergyDifference(index, max_weight_state));
      sum_weight_list_[1] = weight_list_[0];
      
      for (std::int32_t state = 1; state < num_state; ++state) {
         if (state == max_weight_state) {
            weight_list_[state] = std::exp(-beta*system->GetEnergyDifference(index, 0));
         }
         else {
            weight_list_[state] = std::exp(-beta*system->GetEnergyDifference(index, state));
         }
         sum_weight_list_[state + 1] = sum_weight_list_[state] + weight_list_[state];
      }
      sum_weight_list_[0] = sum_weight_list_[num_state];
      
      const std::int32_t temp = system->GetStateNumber(index);
      const std::int32_t now_state = (temp == 0) ? max_weight_state : ((temp == max_weight_state) ? 0 : temp);
      double prob_sum = 0.0;
      const double dist = dist_real_(system->GetRandomNumberEngine());
      for (std::int32_t j = 0; j < num_state; ++j) {
         const double d_ij = sum_weight_list_[now_state + 1] - sum_weight_list_[j] + sum_weight_list_[1];
         prob_sum += std::max(0.0, std::min({d_ij, 1.0 + weight_list_[j] - d_ij, 1.0, weight_list_[j]}));
         if (dist < prob_sum) {
            return (j == max_weight_state) ? 0 : ((j == 0) ? max_weight_state : j);
         }
      }
      return num_state - 1;
   }
   
   template<typename RandType>
   bool DecideAcceptance(RandType *random_number_engine, const double delta_energy, const double beta) {
      return delta_energy <= 0.0 || std::exp(-beta*delta_energy) > dist_real_(*random_number_engine);
   }
   
private:
   const std::int32_t max_num_state_;
   std::uniform_real_distribution<double> dist_real_;
   std::vector<double> weight_list_;
   std::vector<double> sum_weight_list_;

};
   

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
