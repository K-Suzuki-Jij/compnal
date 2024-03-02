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
//  template_system.hpp
//  compnal
//
//  Created by kohei on 2023/05/06.
//  
//

#pragma once

#include "../../../model/utility/variable.hpp"
#include <Eigen/Dense>

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

//! @brief Generate random spin configurations.
//! @tparam ModelType Model class.
//! @tparam RandType Random number engine class.
//! @param model The model.
//! @param random_number_engine Random number engine.
//! @return Random spin configurations.
template<class ModelType, class RandType>
std::vector<model::utility::Spin> GenerateRandomSpins(const ModelType &model,
                                                      RandType *random_number_engine) {
   
   const std::int32_t system_size = model.GetLattice().GetSystemSize();
   const std::vector<std::int32_t> &twice_spin_magnitude = model.GetTwiceSpinMagnitude();
   
   std::vector<model::utility::Spin> spins;
   spins.reserve(system_size);
   
   for (std::int32_t i = 0; i < system_size; ++i) {
      auto spin = model::utility::Spin{0.5*twice_spin_magnitude[i], model.GetSpinScaleFactor()};
      spin.SetStateRandomly(random_number_engine);
      spins.push_back(spin);
   }
   
   return spins;
}

template<class ModelType, class RandType>
class System;

//! @brief Base system class for the Ising model.
//! @tparam ModelType Model class.
//! @tparam RandType Random number engine class.
template<class ModelType, class RandType>
class BaseIsingSystem {
  
public:
   //! @brief Constructor.
   //! @param model The model.
   //! @param seed The seed of the random number engine.
   BaseIsingSystem(const ModelType &model, const typename RandType::result_type seed):
   model_(model),
   system_size_(model.GetLattice().GetSystemSize()),
   bc_(model.GetLattice().GetBoundaryCondition()),
   random_number_engine_(RandType(seed)),
   sample_(GenerateRandomSpins(model, &random_number_engine_)),
   energy_(model.CalculateEnergy(ExtractSample())) {}
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
   //! @brief Extract the sample as Eigen::Vector.
   //! @return The sample.
   Eigen::Vector<typename ModelType::PHQType, Eigen::Dynamic> ExtractSample() const {
      Eigen::Vector<typename ModelType::PHQType, Eigen::Dynamic> sample(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         sample(i) = sample_[i].GetValue();
      }
      return sample;
   }
      
   void SetSampleByValue(const Eigen::Vector<typename ModelType::PHQType, Eigen::Dynamic> &sample_list) {
      if (sample_list.size() != this->sample_.size()) {
         throw std::invalid_argument("The size of initial variables is not equal to the system size.");
      }
      for (std::size_t i = 0; i < this->sample_.size(); ++i) {
         this->sample_[i].SetValue(sample_list(i));
      }
      this->d_E_ = this->model_.GenerateEnergyDifference(this->sample_);
      this->energy_ = this->model_.CalculateEnergy(this->ExtractSample());
   }
   
   //! @brief Get the sample.
   //! @return The sample.
   const std::vector<model::utility::Spin> &GetSample() const {
      return sample_;
   }
   
   //! @brief Get the number of state.
   //! @param index The index of the variable.
   //! @return The number of state.
   std::int32_t GetNumState(const std::int32_t index) const {
      return sample_[index].GetNumState();
   }
   
   //! @brief Get the state number.
   //! @param index The index of the variable.
   //! @return The state number.
   std::int32_t GetStateNumber(const std::int32_t index) const {
      return sample_[index].GetStateNumber();
   }
   
   //! @brief Get the state number, which has the maximum Boltzmann Weight: exp(-beta ΔE).
   //! @param index The index of the variable.
   //! @return The state number.
   std::int32_t GetMaxBoltzmannWeightStateNumber(const std::int32_t index) const {
      if (d_E_[index] >= 0) {
         return 0;
      }
      else {
         return sample_[index].GetNumState() - 1;
      }
   }
   
   //! @brief Get the max number of state.
   //! @return The max number of state.
   std::int32_t GetMaxNumState() const {
      std::int32_t max_num_state = 0;
      for (std::int32_t i = 0; i < system_size_; i++) {
         max_num_state = std::max(max_num_state, this->GetNumState(i));
      }
      return max_num_state;
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
      return (sample_[index].GetValueFromState(candidate_state) - sample_[index].GetValue())*d_E_[index];
   }
   
   //! @brief Get the energy.
   //! @return The energy.
   double GetEnergy() const {
      return energy_;
   }
   
   //! @brief Get random number engine.
   //! @return The random number engine.
   RandType &GetRandomNumberEngine() {
      return random_number_engine_;
   }
   
protected:
   //! @brief The model.
   const ModelType &model_;
   
   //! @brief The system size.
   const std::int32_t system_size_ = 0;

   //! @brief The boundary condition.
   const lattice::BoundaryCondition bc_ = lattice::BoundaryCondition::NONE;

   //! @brief The random number engine.
   RandType random_number_engine_;
   
   //! @brief The spin configuration.
   std::vector<model::utility::Spin> sample_;
   
   //! @brief The energy difference.
   std::vector<double> d_E_;
   
   //! @brief The energy.
   double energy_ = 0.0;

};


} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal
