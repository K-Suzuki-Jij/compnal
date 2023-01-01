//
//  Copyright 2022 Kohei Suzuki
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
//  system_ising_chain.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_CHAIN_HPP_
#define COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_CHAIN_HPP_

#include "../../lattice/all.hpp"
#include "../../model/all.hpp"
#include "base_system.hpp"


namespace compnal {
namespace solver {
namespace cmc_utility {

template<typename RealType>
class CMCSystem<model::Ising<lattice::Chain, RealType>>: public CMCBaseIsingSystem {
   
   using ModelType = model::Ising<lattice::Chain, RealType>;
   
public:
   using ValueType = typename ModelType::ValueType;
   
   CMCSystem(const ModelType &model):
   system_size_(model.GetSystemSize()),
   bc_(model.GetBoundaryCondition()),
   quadratic_(model.GetQuadratic()),
   linear_(model.GetLinear()) {}
   
   void InitializeSSF(const uint64_t seed) {
      sample_ = this->GenerateRandomSpin(seed, system_size_);
      energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   void Flip(const std::int32_t index) {
      if (bc_ == lattice::BoundaryCondition::PBC) {
         if (0 < index && index < system_size_ - 1) {
            energy_difference_[index - 1] += 4*quadratic_*sample_[index - 1]*sample_[index];
            energy_difference_[index] *= -1;
            energy_difference_[index + 1] += 4*quadratic_*sample_[index]*sample_[index + 1];
         }
         else if (index == 0) {
            energy_difference_[0] *= -1;
            energy_difference_[1] += 4*quadratic_*sample_[0]*sample_[1];
            energy_difference_[system_size_ - 1] += 4*quadratic_*sample_[0]*sample_[system_size_ - 1];
         }
         else {
            energy_difference_[0] *= 4*quadratic_*sample_[system_size_ - 1]*sample_[0];
            energy_difference_[system_size_ - 2] += 4*quadratic_*sample_[system_size_ - 2]*sample_[system_size_ - 1];
            energy_difference_[system_size_ - 1] *= -1;
         }
      }
      else if (bc_ == lattice::BoundaryCondition::OBC) {
         if (0 < index && index < system_size_ - 1) {
            energy_difference_[index - 1] += 4*quadratic_*sample_[index - 1]*sample_[index];
            energy_difference_[index] *= -1;
            energy_difference_[index + 1] += 4*quadratic_*sample_[index]*sample_[index + 1];
         }
         else if (index == 0) {
            energy_difference_[0] *= -1;
            energy_difference_[1] += 4*quadratic_*sample_[0]*sample_[1];
         }
         else {
            energy_difference_[system_size_ - 2] += 4*quadratic_*sample_[system_size_ - 2]*sample_[system_size_ - 1];
            energy_difference_[system_size_ - 1] *= -1;
         }
      }
      else {
         throw std::runtime_error("Unsupported BoundaryCondition");
      }
      sample_[index] *= -1;
   }
   
   const std::vector<typename ModelType::OPType> &GetSample() const {
      return sample_;
   }
   
   typename ModelType::ValueType GetEnergyDifference(const std::int32_t index) const {
      return energy_difference_[index];
   }
   
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
private:
   const std::int32_t system_size_;
   const lattice::BoundaryCondition bc_;
   const typename ModelType::QuadraticType quadratic_;
   const typename ModelType::LinearType linear_;
   
   std::vector<typename ModelType::OPType> sample_;
   std::vector<typename ModelType::ValueType> energy_difference_;
   
   std::vector<typename ModelType::ValueType> GenerateEnergyDifference(const std::vector<typename ModelType::OPType> &sample) const {
      std::vector<typename ModelType::ValueType> energy_difference(system_size_);
      if (bc_ == lattice::BoundaryCondition::PBC) {
         for (std::int32_t index = 0; index < system_size_ - 1; ++index) {
            energy_difference[index] += -2*quadratic_*sample[index]*sample[index + 1] - 2*linear_*sample[index];
            energy_difference[index + 1] += -2*quadratic_*sample[index]*sample[index + 1];
         }
         energy_difference[system_size_ - 1] += -2*quadratic_*sample[system_size_ - 1]*sample[0] - 2*linear_*sample[system_size_ - 1];
         energy_difference[0] += -2*quadratic_*sample[system_size_ - 1]*sample[0];
      }
      else if (bc_ == lattice::BoundaryCondition::OBC) {
         for (std::int32_t index = 0; index < system_size_ - 1; ++index) {
            energy_difference[index] += -2*quadratic_*sample[index]*sample[index + 1] - 2*linear_*sample[index];
            energy_difference[index + 1] += -2*quadratic_*sample[index]*sample[index + 1];
         }
         energy_difference[system_size_ - 1] += -2*linear_*sample[system_size_ - 1];
      }
      else {
         throw std::runtime_error("Unsupported BinaryCondition");
      }
      return energy_difference;
   }
   
};

template<typename RealType>
CMCSystem(const model::Ising<lattice::Chain, RealType>) -> CMCSystem<model::Ising<lattice::Chain, RealType>>;

} // namespace cmc_utility
} // namespace solver
} // namespace compnal



#endif /* COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_CHAIN_HPP_ */
