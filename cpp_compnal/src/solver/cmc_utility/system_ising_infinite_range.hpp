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
//  system_ising_infinite_range.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_INFINITE_RANGE_HPP_
#define COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_INFINITE_RANGE_HPP_

#include "../../lattice/all.hpp"
#include "../../model/all.hpp"
#include "base_system.hpp"

namespace compnal {
namespace solver {
namespace cmc_utility {

template<typename RealType>
class CMCSystem<model::Ising<lattice::InfiniteRange, RealType>>: public CMCBaseIsingSystem {
   
   using ModelType = model::Ising<lattice::InfiniteRange, RealType>;
   
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
      const auto spin = sample_[index];
      for (std::int32_t i = 0; i < index; ++i) {
         energy_difference_[i] += 4*quadratic_*sample_[i]*spin;
      }
      for (std::int32_t i = index + 1; i < system_size_; ++i) {
         energy_difference_[i] += 4*quadratic_*sample_[i]*spin;
      }
      energy_difference_[index] *= -1;
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
      for (std::int32_t i = 0; i < system_size_; ++i) {
         energy_difference[i] += -2*linear_*sample[i];
         for (std::int32_t j = i + 1; j < system_size_; ++j) {
            energy_difference[i] += -2*quadratic_*sample[i]*sample[j];
            energy_difference[j] += -2*quadratic_*sample[i]*sample[j];
         }
      }
      return energy_difference;
   }
   
};

template<typename RealType>
CMCSystem(const model::Ising<lattice::InfiniteRange, RealType>) -> CMCSystem<model::Ising<lattice::InfiniteRange, RealType>>;

} // namespace cmc_utility
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_INFINITE_RANGE_HPP_ */
