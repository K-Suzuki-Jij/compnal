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
//  ising_chain.hpp
//  compnal
//
//  Created by kohei on 2023/05/06.
//  
//

#pragma once

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
class System<model::classical::Ising<lattice::Chain>, RandType>: public BaseIsingSystem<model::classical::Ising<lattice::Chain>, RandType> {
   
   //! @brief Model type.
   using ModelType = model::classical::Ising<lattice::Chain>;
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed),
   linear_(model.GetLinear()),
   quadratic_(model.GetQuadratic()) {
      this->d_E_ = model.GenerateEnergyDifference(this->sample_);
   }
      
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
      if (update_state == this->sample_[index].GetStateNumber()) {
         return;
      }
      this->energy_ += this->GetEnergyDifference(index, update_state);
      const double diff = this->quadratic_*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
      if (this->bc_ == lattice::BoundaryCondition::PBC) {
         this->d_E_[(index - 1 + this->system_size_)%this->system_size_] += diff;
         this->d_E_[(index + 1)%this->system_size_] += diff;
      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {
         if (index < this->system_size_ - 1) {
            this->d_E_[index + 1] += diff;
         }
         if (index > 0) {
            this->d_E_[index - 1] += diff;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
private:
   //! @brief The linear interaction.
   const double linear_ = 0;

   //! @brief The quadratic interaction.
   const double quadratic_ = 0;
   
};

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
