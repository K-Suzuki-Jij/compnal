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
//  system_poly_ising_any_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_UTILITY_CMC_SYSTEM_POLY_ISING_ANY_LATTICE_HPP_
#define COMPNAL_SOLVER_UTILITY_CMC_SYSTEM_POLY_ISING_ANY_LATTICE_HPP_

#include "../../lattice/all.hpp"
#include "../../model/all.hpp"
#include "base_system.hpp"

namespace compnal {
namespace solver {
namespace utility_cmc {

template<typename RealType>
class CMCSystem<model::classical::PolynomialIsing<lattice::AnyLattice, RealType>>: public CMCBaseIsingSystem {
   
   using ModelType = model::classical::PolynomialIsing<lattice::AnyLattice, RealType>;
   
public:
   using ValueType = typename ModelType::ValueType;
   
   CMCSystem(const ModelType &model):
   system_size_(model.GetSystemSize()),
   bc_(model.GetLattice().GetBoundaryCondition()),
   key_value_list_(model.GetKeyValueList()),
   adjacency_list_(model.GetAdjacencyList()) {}
   
   void InitializeSSF(const uint64_t seed) {
      sample_ = this->GenerateRandomSpin(seed, system_size_);
      sign_list_ = GenerateSignList(sample_);
      energy_difference_ = GenerateEnergyDifference(sample_);
   }
   
   void Flip(const std::int32_t index) {
      for (const auto &interaction_index: adjacency_list_[index]) {
         const std::vector<std::int32_t> &key_list = key_value_list_[interaction_index].first;
         const ValueType value = 4*sign_list_[interaction_index]*key_value_list_[interaction_index].second;
         sign_list_[interaction_index] *= -1;
         for (const auto &update_index: key_list) {
            if (update_index == index) continue;
            energy_difference_[update_index] += value;
         }
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
   const std::vector<std::pair<std::vector<std::int32_t>, ValueType>> &key_value_list_;
   const std::vector<std::vector<std::size_t>> &adjacency_list_;

   std::vector<typename ModelType::OPType> sample_;
   std::vector<typename ModelType::ValueType> energy_difference_;
   std::vector<std::int8_t> sign_list_;
   
   std::vector<typename ModelType::ValueType> GenerateEnergyDifference(const std::vector<typename ModelType::OPType> &sample) const {
      std::vector<typename ModelType::ValueType> energy_difference(system_size_);
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         const std::vector<std::int32_t> &key_list = key_value_list_[i].first;
         const ValueType value = -2*key_value_list_[i].second*sign_list_[i];
         for (const auto &index: key_list) {
            energy_difference[index] += value;
         }
      }
      return energy_difference;
   }
   
   std::vector<std::int8_t> GenerateSignList(const std::vector<typename ModelType::OPType> &sample) const {
      std::vector<std::int8_t> sign_list(key_value_list_.size());
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         std::int8_t sign = 1;
         for (const auto &index: key_value_list_[i].first) {
            sign *= sample[index];
         }
         sign_list[i] = sign;
      }
      return sign_list;
   }
   
};

template<typename RealType>
CMCSystem(const model::classical::PolynomialIsing<lattice::AnyLattice, RealType>) -> CMCSystem<model::classical::PolynomialIsing<lattice::AnyLattice, RealType>>;


} // namespace utility_cmc
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_UTILITY_CMC_SYSTEM_POLY_ISING_ANY_LATTICE_HPP_ */
