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
//  ising_chain.hpp
//  compnal
//
//  Created by kohei on 2023/05/06.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CHAIN_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CHAIN_HPP_

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
class System<model::classical::Ising<lattice::Chain>, RandType> {
   //! @brief Model type.
   using ModelType = model::classical::Ising<lattice::Chain>;
   
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   System(const ModelType &model, const typename RandType::result_type seed):
   system_size_(model.GetLattice().GetSystemSize()),
   bc_(model.GetLattice().GetBoundaryCondition()),
   linear_(model.GetLinear()),
   quadratic_(model.GetQuadratic()),
   random_number_engine_(RandType(seed)) {
      sample_ = GenerateRandomSpins(model, &random_number_engine_);
      base_energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   //! @brief Set sample by states.
   //! Here, the states represents energy levels. For example for S=1/2 ising spins,
   //! s=-1/2 corresponds to the state being 0 and s=1/2 corresponds to the state being 1.
   //! @param state_list The list of states.
   void SetSampleByState(const std::vector<std::int32_t> &state_list) {
      if (state_list.size() != sample_.size()) {
         throw std::runtime_error("The size of initial variables is not equal to the system size.");
      }
      for (std::size_t i = 0; i < sample_.size(); ++i) {
         sample_[i].SetState(state_list[i]);
      }
      base_energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   //! @brief Flip a variable.
   //! @param index The index of the variable to be flipped.
   //! @param update_state The state number to be updated.
   void Flip(const std::int32_t index, const std::int32_t update_state) {
      const double diff = quadratic_*(sample_[index].GetValueFromState(update_state) - sample_[index].GetValue());
      if (bc_ == lattice::BoundaryCondition::PBC) {
         if (0 < index && index < system_size_ - 1) {
            base_energy_difference_[index - 1] += diff;
            base_energy_difference_[index + 1] += diff;
         }
         else if (index == 0) {
            base_energy_difference_[1] += diff;
            base_energy_difference_[system_size_ - 1] += diff;
         }
         else {
            base_energy_difference_[0] += diff;
            base_energy_difference_[system_size_ - 2] += diff;
         }
      }
      else if (bc_ == lattice::BoundaryCondition::OBC) {
         if (0 < index && index < system_size_ - 1) {
            base_energy_difference_[index - 1] += diff;
            base_energy_difference_[index + 1] += diff;
         }
         else if (index == 0) {
            base_energy_difference_[1] += diff;
         }
         else {
            base_energy_difference_[system_size_ - 2] += diff;
         }
      }
      else {
         throw std::runtime_error("Unsupported BoundaryCondition");
      }
      sample_[index].SetState(update_state);
   }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
   //! @brief Extract the sample.
   //! @return The sample.
   std::vector<typename ModelType::PHQType> ExtractSample() const {
      std::vector<typename ModelType::PHQType> sample(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         sample[i] = sample_[i].GetValue();
      }
      return sample;
   }
   
   //! @brief Generate candidate state.
   //! @param index The index of the variable.
   //! @return The candidate state.
   std::int32_t GenerateCandidateState(const std::int32_t index) {
      return sample_[index].GenerateCandidateState(&random_number_engine_);
   }
   
   //! @brief Get the energy difference.
   //! @param index The index of the variable.
   //! @param candidate_state The candidate state.
   //! @return The energy difference.
   double GetEnergyDifference(const std::int32_t index, const std::int32_t candidate_state) const {
      return (sample_[index].GetValueFromState(candidate_state) - sample_[index].GetValue())*base_energy_difference_[index];
   }
   
private:
   //! @brief The system size.
   const std::int32_t system_size_ = 0;

   //! @brief The boundary condition.
   const lattice::BoundaryCondition bc_ = lattice::BoundaryCondition::NONE;

   //! @brief The linear interaction.
   const double linear_ = 0;

   //! @brief The quadratic interaction.
   const double quadratic_ = 0;

   //! @brief The spin configuration.
   std::vector<model::utility::Spin> sample_;

   //! @brief The random number engine.
   RandType random_number_engine_;

   //! @brief The energy difference.
   std::vector<double> base_energy_difference_;
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const std::vector<model::utility::Spin> &sample) const {
      std::vector<double> base_energy_difference(system_size_);
      if (bc_ == lattice::BoundaryCondition::PBC) {
         for (std::int32_t index = 0; index < system_size_ - 1; ++index) {
            base_energy_difference[index] += quadratic_*sample[index + 1].GetValue() + linear_;
            base_energy_difference[index + 1] += quadratic_*sample[index].GetValue();
         }
         base_energy_difference[system_size_ - 1] += quadratic_*sample[0].GetValue() + linear_;
         base_energy_difference[0] += quadratic_*sample[system_size_ - 1].GetValue();
      }
      else if (bc_ == lattice::BoundaryCondition::OBC) {
         for (std::int32_t index = 0; index < system_size_ - 1; ++index) {
            base_energy_difference[index] += quadratic_*sample[index + 1].GetValue() + linear_;
            base_energy_difference[index + 1] += quadratic_*sample[index].GetValue();
         }
         base_energy_difference[system_size_ - 1] += linear_;
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


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CHAIN_HPP_ */
