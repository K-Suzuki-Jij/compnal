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
//  poly_ising_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2023/06/29.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_INFINITE_RANGE_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_INFINITE_RANGE_HPP_

#include "template_system.hpp"
#include "../../../utility/combination.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/infinite_range.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief System class for the Ising model on a chain.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::PolynomialIsing<lattice::InfiniteRange>, RandType>: public BaseIsingSystem<model::classical::PolynomialIsing<lattice::InfiniteRange>, RandType> {
   
   //! @brief Model type.
   using ModelType = model::classical::PolynomialIsing<lattice::InfiniteRange>;
   
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   interaction_(model.GetInteraction()),
   degree_(model.GetDegree()) {
      this->d_E = GenerateEnergyDifference(this->sample_);
   }
   
   //! @brief Set sample by states.
   //! Here, the states represents energy levels. For example for S=1/2 ising spins,
   //! s=-1/2 corresponds to the state being 0 and s=1/2 corresponds to the state being 1.
   //! @param state_list The list of states.
   void SetSampleByState(const std::vector<std::int32_t> &state_list) {
      if (state_list.size() != this->sample_.size()) {
         throw std::invalid_argument("The size of initial variables is not equal to the system size.");
      }
      for (std::size_t i = 0; i < this->sample_.size(); ++i) {
         this->sample_[i].SetState(state_list[i]);
      }
      this->d_E = GenerateEnergyDifference(this->sample_);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
      FlipAny(index, update_state);      
   }
   
   
private:
   //! @brief The polynomial interaction.
   std::unordered_map<std::int32_t, double> interaction_;
   
   //! @brief The degree of the interactions.
   std::int32_t degree_ = 0;
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const std::vector<model::utility::Spin> &sample) const {
      std::vector<double> d_E(this->system_size_);
      
      for (std::int32_t index = 0; index < this->system_size_; ++index) {
         double val = 0.0;
         for (const auto &it: interaction_) {
            if (it.first < 2) {
               if (it.first == 1) {
                  val += it.second;
               }
               continue;
            }
            std::vector<std::int32_t> indices(it.first - 1);
            std::int32_t start_index = 0;
            std::int32_t size = 0;
            
            while (true) {
               for (std::int32_t i = start_index; i < this->system_size_ - 1; ++i) {
                  indices[size++] = i;
                  if (size == it.first - 1) {
                     double spin_prod = 1;
                     for (std::int32_t j = 0; j < it.first - 1; ++j) {
                        if (indices[j] >= index) {
                           spin_prod *= sample[indices[j] + 1].GetValue();
                        }
                        else {
                           spin_prod *= sample[indices[j]].GetValue();
                        }
                     }
                     val += it.second*spin_prod;
                     break;
                  }
               }
               --size;
               if (size < 0) {
                  break;
               }
               start_index = indices[size] + 1;
            }
         }
         d_E[index] += val;
      }      
      return d_E;
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void FlipAny(const std::int32_t index, const std::int32_t update_state) {
      for (const auto &it: interaction_) {
         if (it.first <= 1) {
            continue;
         }
         const double val = it.second*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
         std::vector<std::int32_t> indices(it.first);
         std::int32_t start_index = 0;
         std::int32_t size = 0;
         
         while (true) {
            for (std::int32_t i = start_index; i < this->system_size_ - 1; ++i) {
               indices[size++] = i;
               if (size == it.first - 1) {
                  double spin_prod = 1;
                  for (std::int32_t j = 0; j < it.first - 1; ++j) {
                     if (indices[j] >= index) {
                        spin_prod *= this->sample_[indices[j] + 1].GetValue();
                     }
                     else {
                        spin_prod *= this->sample_[indices[j]].GetValue();
                     }
                  }

                  for (std::int32_t j = 0; j < it.first - 1; ++j) {
                     if (indices[j] >= index) {
                        this->d_E[indices[j] + 1] += val*spin_prod/this->sample_[indices[j] + 1].GetValue();
                     }
                     else {
                        this->d_E[indices[j]] += val*spin_prod/this->sample_[indices[j]].GetValue();
                     }
                  }
                  break;
               }
            }
            --size;
            if (size < 0) {
               break;
            }
            start_index = indices[size] + 1;
         }
      }
      this->sample_[index].SetState(update_state);
   }
   
};

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_INFINITE_RANGE_HPP_ */
