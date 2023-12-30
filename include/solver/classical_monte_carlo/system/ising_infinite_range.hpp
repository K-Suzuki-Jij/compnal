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
//  ising_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2023/06/14.
//  
//

#pragma once

#include "template_system.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/infinite_range.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief System class for the Ising model on a square.
//! @tparam RandType Random number engine class.
template<class RandType>
class System<model::classical::Ising<lattice::InfiniteRange>, RandType>: public BaseIsingSystem<model::classical::Ising<lattice::InfiniteRange>, RandType> {
   //! @brief Model type.
   using ModelType = model::classical::Ising<lattice::InfiniteRange>;
   
   
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
      this->energy_ += this->GetEnergyDifference(index, update_state);
      const double diff = this->quadratic_*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
      for (std::int32_t i = 0; i < index; ++i) {
         this->d_E_[i] += diff;
      }
      for (std::int32_t i = index + 1; i < this->system_size_; ++i) {
         this->d_E_[i] += diff;
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
