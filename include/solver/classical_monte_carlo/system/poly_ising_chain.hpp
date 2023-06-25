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
//  poly_ising_chain.hpp
//  compnal
//
//  Created by kohei on 2023/06/25.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_CHAIN_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_CHAIN_HPP_

#include "template_system.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/chain.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief System class for the Ising model on a chain.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::PolynomialIsing<lattice::Chain>, RandType>: public BaseIsingSystem<model::classical::PolynomialIsing<lattice::Chain>, RandType> {
  
   //! @brief Model type.
   using ModelType = model::classical::PolynomialIsing<lattice::Chain>;
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   interaction_(model.GetInteraction()),
   degree_(model.GetDegree()) {
      this->base_energy_difference_ = GenerateEnergyDifference(this->sample_);
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
      this->base_energy_difference_ = GenerateEnergyDifference(this->sample_);
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
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         for (std::int32_t index = 0; index < this->system_size_; ++index) {
            double val = 0;
            for (const auto &it: interaction_) {
               double spin_prod = 1;
               for (std::int32_t diff = 1; diff < it.first; ++diff) {
                  spin_prod *= this->sample_[(index - diff + this->system_size_)%this->system_size_].GetValue();
               }
               for (std::int32_t diff = it.first - 1; diff >= 0; --diff) {
                  val += it.second*spin_prod;
                  const std::int32_t ind_1 = (index - diff + this->system_size_)%this->system_size_;
                  const std::int32_t ind_2 = (index - diff + it.first + this->system_size_)%this->system_size_;
                  spin_prod = spin_prod*this->sample_[ind_2].GetValue()/this->sample_[ind_1].GetValue();
               }
            }
            d_E[index] += val;
         }
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         for (std::int32_t index = 0; index < this->system_size_; ++index) {
            double val = 0;
            for (const auto &it: interaction_) {
               double spin_prod = 1;
               for (std::int32_t diff = 1; diff < it.first; ++diff) {
                  spin_prod *= this->sample_[(index - diff + this->system_size_)%this->system_size_].GetValue();
               }
               for (std::int32_t diff = it.first - 1; diff >= 0; --diff) {
                  if (index - diff >= 0 && index - diff + it.first - 1 < this->system_size_) {
                     val += it.second*spin_prod;
                  }
                  const std::int32_t ind_1 = (index - diff + this->system_size_)%this->system_size_;
                  const std::int32_t ind_2 = (index - diff + it.first + this->system_size_)%this->system_size_;
                  spin_prod = spin_prod*this->sample_[ind_2].GetValue()/this->sample_[ind_1].GetValue();
               }
            }
            d_E[index] += val;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      return d_E;
   }
   
};



} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_POLY_ISING_CHAIN_HPP_ */
