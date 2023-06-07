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
//  ising_square.hpp
//  compnal
//
//  Created by kohei on 2023/06/08.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_SQUARE_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_SQUARE_HPP_

#include "template_system.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../model/utility/variable.hpp"
#include "../../../lattice/square.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

template<class RandType>
class System<model::classical::Ising<lattice::Square>, RandType>: public BaseIsingSystem<model::classical::Ising<lattice::Square>, RandType> {
   //! @brief Model type.
   using ModelType = model::classical::Ising<lattice::Square>;
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   BaseIsingSystem<ModelType, RandType>::BaseIsingSystem(model, seed) {
      this->base_energy_difference_ = GenerateEnergyDifference(this->sample_);
   }
   
   //! @brief Set sample by states.
   //! Here, the states represents energy levels. For example for S=1/2 ising spins,
   //! s=-1/2 corresponds to the state being 0 and s=1/2 corresponds to the state being 1.
   //! @param state_list The list of states.
   void SetSampleByState(const std::vector<std::int32_t> &state_list) {
      if (state_list.size() != this->sample_.size()) {
         throw std::runtime_error("The size of initial variables is not equal to the system size.");
      }
      for (std::size_t i = 0; i < this->sample_.size(); ++i) {
         this->sample_[i].SetState(state_list[i]);
      }
      this->base_energy_difference_ = GenerateEnergyDifference(this->sample_);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
      const double diff = this->quadratic_*(this->sample_[index].GetValueFromState(update_state) - this->sample_[index].GetValue());
      if (this->bc_ == lattice::BoundaryCondition::PBC) {

      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {

      }
      else {
         throw std::runtime_error("Unsupported BoundaryCondition");
      }
      this->sample_[index].SetState(update_state);
   }
   
private:
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const std::vector<model::utility::Spin> &sample) const {
      std::vector<double> base_energy_difference(this->system_size_);
      if (this->bc_ == lattice::BoundaryCondition::PBC) {

      }
      else if (this->bc_ == lattice::BoundaryCondition::OBC) {

      }
      else {
         throw std::runtime_error("Unsupported BinaryCondition");
      }
      return base_energy_difference;
   }
   
};



} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_SQUARE_HPP_ */
