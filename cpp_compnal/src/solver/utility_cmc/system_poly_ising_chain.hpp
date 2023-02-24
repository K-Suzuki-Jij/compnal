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
//  system_poly_ising_chain.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_UTILITY_CMC_SYSTEM_POLY_ISING_CHAIN_HPP_
#define COMPNAL_SOLVER_UTILITY_CMC_SYSTEM_POLY_ISING_CHAIN_HPP_

#include "../../lattice/all.hpp"
#include "../../model/all.hpp"
#include "base_system.hpp"


namespace compnal {
namespace solver {
namespace utility_cmc {

template<typename RealType>
class CMCSystem<model::classical::PolynomialIsing<lattice::Chain, RealType>>: public CMCBaseIsingSystem {
   
   using ModelType = model::classical::PolynomialIsing<lattice::Chain, RealType>;
   
   using PolynomialType = typename ModelType::PolynomialType;
   
   using OPType = typename ModelType::OPType;

public:
   using ValueType = typename ModelType::ValueType;
   
   CMCSystem(const ModelType &model):
   system_size_(model.GetSystemSize()),
   bc_(model.GetLattice().GetBoundaryCondition()),
   interaction_(model.GetInteraction()) {}
   
   void InitializeSSF(const uint64_t seed) {
      sample_ = this->GenerateRandomSpin(seed, system_size_);
      energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   void Flip(const std::int32_t index) {
      if (bc_ == lattice::BoundaryCondition::PBC) {
         for (const auto &it: interaction_) {
            const ValueType target_ineraction = it.second;
            for (std::int32_t i = 0; i < it.first; ++i) {
               OPType sign = 1;
               for (std::int32_t j = 0; j < it.first; ++j) {
                  std::int32_t connected_index = index - it.first + 1 + i + j;
                  if (connected_index < 0) {
                     connected_index += system_size_;
                  }
                  else if (connected_index >= system_size_) {
                     connected_index -= system_size_;
                  }
                  sign *= sample_[connected_index];
               }
               for (std::int32_t j = 0; j < it.first; ++j) {
                  std::int32_t connected_index = index - it.first + 1 + i + j;
                  if (connected_index < 0) {
                     connected_index += system_size_;
                  }
                  else if (connected_index >= system_size_) {
                     connected_index -= system_size_;
                  }
                  if (connected_index != index) {
                     energy_difference_[connected_index] += 4*target_ineraction*sign;
                  }
               }
            }
         }
      }
      else if (bc_ == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            const ValueType target_ineraction = it.second;
            
            for (std::int32_t i = std::max(index - it.first + 1, 0); i <= index; ++i) {
               if (i > system_size_ - it.first) {
                  break;
               }
               OPType sign = 1;
               for (std::int32_t j = i; j < i + it.first; ++j) {
                  sign *= sample_[j];
               }
               for (std::int32_t j = i; j < it.first; ++j) {
                  energy_difference_[j] += 4*target_ineraction*sign;
               }
               for (std::int32_t j = index + 1; j < i + it.first; ++j) {
                  energy_difference_[j] += 4*target_ineraction*sign;
               }
            }
         }
      }
      else {
         throw std::runtime_error("Unsupported BinaryCondition");
      }
      energy_difference_[index] *= -1;
      sample_[index] *= -1;
   }
   
   const std::vector<OPType> &GetSample() const {
      return sample_;
   }
   
   ValueType GetEnergyDifference(const std::int32_t index) const {
      return energy_difference_[index];
   }
   
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
private:
   const std::int32_t system_size_;
   const lattice::BoundaryCondition bc_;
   const PolynomialType interaction_;
   
   std::vector<OPType> sample_;
   std::vector<ValueType> energy_difference_;
   
   std::vector<ValueType> GenerateEnergyDifference(const std::vector<OPType> &sample) const {
      std::vector<ValueType> energy_difference(system_size_);
      if (bc_ == lattice::BoundaryCondition::PBC) {
         for (const auto &it: interaction_) {
            const ValueType target_ineraction = it.second;
            for (std::int32_t index = 0; index < system_size_; ++index) {
               ValueType val = 0;
               for (std::int32_t i = 0; i < it.first; ++i) {
                  OPType sign = 1;
                  for (std::int32_t j = 0; j < it.first; ++j) {
                     std::int32_t connected_index = index - it.first + 1 + i + j;
                     if (connected_index < 0) {
                        connected_index += system_size_;
                     }
                     else if (connected_index >= system_size_) {
                        connected_index -= system_size_;
                     }
                     sign *= sample[connected_index];
                  }
                  val += sign*target_ineraction;
               }
               energy_difference[index] = -2.0*val;
            }
         }
      }
      else if (bc_ == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            const ValueType target_ineraction = it.second;
            for (std::int32_t index = 0; index < system_size_; ++index) {
               ValueType val = 0;
               for (std::int32_t i = 0; i < it.first; ++i) {
                  if (index - it.first + 1 + i < 0 || index + i >= system_size_) {
                     continue;
                  }
                  OPType sign = 1;
                  for (std::int32_t j = 0; j < it.first; ++j) {
                     std::int32_t connected_index = index - it.first + 1 + i + j;
                     sign *= (sample)[connected_index];
                  }
                  val += sign*target_ineraction;
               }
               energy_difference[index] = -2.0*val;
            }
         }
      }
      else {
         throw std::runtime_error("Unsupported BinaryCondition");
      }
      return energy_difference;
   }
   
};

template<typename RealType>
CMCSystem(const model::classical::PolynomialIsing<lattice::Chain, RealType>) -> CMCSystem<model::classical::PolynomialIsing<lattice::Chain, RealType>>;

} // namespace utility_cmc
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_UTILITY_CMC_SYSTEM_POLY_ISING_CHAIN_HPP_ */
