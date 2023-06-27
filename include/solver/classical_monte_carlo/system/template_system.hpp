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
//  template_system.hpp
//  compnal
//
//  Created by kohei on 2023/05/06.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_

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
   system_size_(model.GetLattice().GetSystemSize()),
   bc_(model.GetLattice().GetBoundaryCondition()),
   random_number_engine_(RandType(seed)),
   sample_(GenerateRandomSpins(model, &random_number_engine_)) {}
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
   //! @brief Extract the sample.
   //! @return The sample.
   Eigen::Vector<typename ModelType::PHQType, Eigen::Dynamic> ExtractSample() const {
      Eigen::Vector<typename ModelType::PHQType, Eigen::Dynamic> sample(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         sample(i) = sample_[i].GetValue();
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
      return (sample_[index].GetValueFromState(candidate_state) - sample_[index].GetValue())*d_E[index];
   }
   
protected:
   //! @brief The system size.
   const std::int32_t system_size_ = 0;

   //! @brief The boundary condition.
   const lattice::BoundaryCondition bc_ = lattice::BoundaryCondition::NONE;

   //! @brief The random number engine.
   RandType random_number_engine_;
   
   //! @brief The spin configuration.
   std::vector<model::utility::Spin> sample_;
   
   //! @brief The energy difference.
   std::vector<double> d_E;

};


} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_TEMPLATE_SYSTEM_HPP_ */
