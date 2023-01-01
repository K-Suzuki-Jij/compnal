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
//  system_ising_any_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_ANY_LATTICE_HPP_
#define COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_ANY_LATTICE_HPP_

#include "../../lattice/all.hpp"
#include "../../model/all.hpp"
#include "base_system.hpp"

namespace compnal {
namespace solver {
namespace cmc_utility {

template<typename RealType>
class CMCSystem<model::Ising<lattice::AnyLattice, RealType>>: public CMCBaseIsingSystem {
   
   using ModelType = model::Ising<lattice::AnyLattice, RealType>;
   
public:
   using ValueType = typename ModelType::ValueType;
   
   CMCSystem(const ModelType &model):
   system_size_(model.GetSystemSize()),
   bc_(model.GetBoundaryCondition()),
   col_ptr_(model.GetColPtr()),
   val_ptr_(model.GetValPtr()),
   row_ptr_(model.GetRowPtr()),
   linear_(model.GetLinear()){}
   
   void InitializeSSF(const uint64_t seed) {
      sample_ = this->GenerateRandomSpin(seed, system_size_);
      energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   void Flip(const std::int32_t index) {
      const auto spin = sample_[index];
      sample_[index] *= -1;
      energy_difference_[index] *= -1;
      for (std::int64_t i = row_ptr_[index]; i < row_ptr_[index + 1]; ++i) {
         energy_difference_[col_ptr_[i]] += 4*val_ptr_[i]*sample_[col_ptr_[i]]*spin;
      }
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
   const std::vector<std::int32_t> &col_ptr_;
   const std::vector<typename ModelType::ValueType> &val_ptr_;
   const std::vector<std::int64_t> &row_ptr_;
   const std::vector<typename ModelType::ValueType> &linear_;

   std::vector<typename ModelType::OPType> sample_;
   std::vector<typename ModelType::ValueType> energy_difference_;
   
   std::vector<typename ModelType::ValueType> GenerateEnergyDifference(const std::vector<typename ModelType::OPType> &sample) const {
      std::vector<typename ModelType::ValueType> energy_difference(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         const auto spin = sample[i];
         for (std::int64_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
            energy_difference[col_ptr_[j]] += -2*val_ptr_[j]*sample[col_ptr_[j]]*spin;
         }
         energy_difference[i] += - 2*linear_[i]*sample[i];
      }
      return energy_difference;
   }
   
};

template<typename RealType>
CMCSystem(const model::Ising<lattice::AnyLattice, RealType>) -> CMCSystem<model::Ising<lattice::AnyLattice, RealType>>;


} // namespace cmc_utility
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CMC_UTILITY_SYSTEM_ISING_ANY_LATTICE_HPP_ */
