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
//  Created by Kohei Suzuki on 2022/06/10.
//

#ifndef COMPNAL_MODEL_POLYNOMIAL_ISING_HPP_
#define COMPNAL_MODEL_POLYNOMIAL_ISING_HPP_

#include "../../lattice/all.hpp"
#include "../../interaction/polynomial_any_interaction.hpp"
#include "../../utility/type.hpp"
#include <vector>
#include <unordered_map>

namespace compnal {
namespace model {

//! @brief Polynomial Ising class.
template<class LatticeType, typename RealType>
class PolynomialIsing {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   
   using ValueType = RealType;
   using IndexType = std::int32_t;
   using OPType = utility::SpinType;
   using PolynomialType = std::unordered_map<std::int32_t, RealType>;
      
   PolynomialIsing(const LatticeType &lattice,
                   const PolynomialType &interaction):
   lattice_(lattice) {
      for (const auto &it: interaction) {
         if (interaction_.size() <= it.first) {
            interaction_.resize(it.first + 1);
         }
         interaction_[it.first] = it.second;
      }
   }

   const std::vector<RealType> &GetInteraction() const {
      return interaction_;
   }
         
   std::int32_t GetSystemSize() const {
      return lattice_.GetSystemSize();
   }
   
   lattice::BoundaryCondition GetBoundaryCondition() const {
      return lattice_.GetBoundaryCondition();
   }
   
   auto GenerateIndexList() const {
      return lattice_.GenerateIndexList();
   }
   
   std::int32_t GetDegree() const {
      return static_cast<std::int32_t>(interaction_.size()) - 1;
   }
   
   const LatticeType &GetLattice() const {
      return lattice_;
   }
   
   RealType CalculateEnergy(const std::vector<OPType> &sample) const {
      return CalculateEnergy(lattice_, sample);
   }
   
   RealType CalculateMoment(const std::vector<std::vector<OPType>> &samples,
                            const std::int32_t degree,
                            const std::int32_t num_threads = utility::DEFAULT_NUM_THREADS) const {
      if (degree <= 0) {
         throw std::runtime_error("degree must be lager than 0.");
      }
      
      RealType val = 0;
#pragma omp parallel for schedule(guided) reduction(+: val) num_threads(num_threads)
      for (std::int32_t i = 0; i < static_cast<std::int32_t>(samples.size()); ++i) {
         RealType avg = CalculateMagnetization(samples[i]);
         RealType prod = 1;
         for (std::int32_t j = 0; j < degree; ++j) {
            prod = prod*avg;
         }
         val += prod;
      }
      return val/samples.size();
   }
   
   RealType CalculateOnsiteAverage(const std::vector<std::vector<OPType>> &samples,
                                         const IndexType index) const {
      if (index < 0) {
         throw std::runtime_error("The index is out of range.");
      }
      RealType val = 0;
      for (std::size_t i = 0; i < samples.size(); ++i) {
         val += samples[i][index];
      }
      return val/samples.size();
   }
   
   RealType CalculateCorrelation(const std::vector<std::vector<OPType>> &samples,
                                 const IndexType ind1,
                                 const IndexType ind2) const {
      const std::int32_t size = static_cast<std::int32_t>(samples.size());
      if (ind1 < 0 || ind2 < 0) {
         throw std::runtime_error("The index is out of range.");
      }
      RealType val = 0;
      for (std::int32_t i = 0; i < size; ++i) {
         val += samples[i][ind1]*samples[i][ind2];
      }
      return val/size;
   }
   
private:
   LatticeType lattice_;
   std::vector<RealType> interaction_;

   
   RealType CalculateEnergy(const lattice::Chain &chain_lattice,
                            const std::vector<OPType> &sample) const {
      RealType energy = 0;
      return energy;
   }
   
   RealType CalculateEnergy(const lattice::Square &square_lattice,
                            const std::vector<OPType> &sample) const {
      RealType energy = 0;
      return energy;
   }
   
   RealType CalculateEnergy(const lattice::Cubic &cubic_lattice,
                            const std::vector<OPType> &sample) const {
      RealType energy = 0;
      return energy;
   }
   
   RealType CalculateEnergy(const lattice::InfiniteRange &infinite_range_lattice,
                            const std::vector<OPType> &sample) const {
      RealType energy = 0;
      return energy;
   }
   
   RealType CalculateMagnetization(const std::vector<OPType> &sample) const {
      RealType val = 0;
      for (std::size_t i = 0; i < sample.size(); ++i) {
         val += sample[i];
      }
      return val/sample.size();
   }
   
};


template<typename RealType>
class PolynomialIsing<lattice::AnyLattice, RealType> {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
  
public:
   using ValueType = RealType;
   using OPType = utility::SpinType;
   using IndexType = typename interaction::PolynomialAnyInteraction<RealType>::IndexType;
   using IndexHash = typename interaction::PolynomialAnyInteraction<RealType>::IndexHash;
   using PolynomialType = typename interaction::PolynomialAnyInteraction<RealType>::PolynomialType;
      
   PolynomialIsing(const lattice::AnyLattice &lattice,
                   const PolynomialType &interaction):
   lattice_(lattice), interaction_(interaction) {}
   
   std::pair<std::vector<std::vector<IndexType>>, std::vector<RealType>> GenerateInteractionAsPair() const {
      return interaction_.GenerateInteractionAsPair();
   }
   
   const std::vector<std::vector<std::int32_t>> &GetKeyList() const {
      return interaction_.GetKeyList();
   }
   
   const std::vector<RealType> &GetValueList() const {
      return interaction_.GetValueList();
   }
   
   const std::vector<IndexType> &GenerateIndexList() const {
      return interaction_.GetIndexList();
   }
   
   const std::unordered_map<IndexType, std::int32_t, IndexHash> &GetIndexMap() const {
      return interaction_.GetIndexMap();
   }
   
   const std::vector<std::vector<std::size_t>> &GetAdjacencyList() const {
      return interaction_.GetAdjacencyList();
   }
   
   std::int32_t GetSystemSize() const {
      return interaction_.GetSystemSize();
   }
   
   std::int32_t GetDegree() const {
      return interaction_.GetDegree();
   }
   
   lattice::BoundaryCondition GetBoundaryCondition() const {
      return lattice_.GetBoundaryCondition();
   }
   
   RealType CalculateEnergy(const std::vector<OPType> &sample) const {
      if (sample.size() != interaction_.GetSystemSize()) {
         throw std::runtime_error("The sample size is not equal to the system size");
      }
      const auto &key_list = interaction_.GetKeyList();
      const auto &value_list = interaction_.GetValueList();
      RealType val = 0;
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         OPType spin = 1;
         for (const auto &index: key_list[i]) {
            spin *= sample[index];
         }
         val += spin*value_list[i];
      }
      return val;
   }
   
   RealType CalculateOnsiteAverage(const std::vector<std::vector<OPType>> &samples,
                                         const IndexType index) const {
      const std::unordered_map<IndexType, std::int32_t, IndexHash> &index_map = interaction_.GetIndexMap();
      if (index_map.count(index) == 0) {
         throw std::runtime_error("The index is out of range.");
      }
      RealType val = 0;
      for (std::size_t i = 0; i < samples.size(); ++i) {
         val += samples[i][index_map.at(index)];
      }
      return val/samples.size();
   }
   
   RealType CalculateMoment(const std::vector<std::vector<OPType>> &samples,
                            const std::int32_t degree,
                            const std::int32_t num_threads = utility::DEFAULT_NUM_THREADS) const {
      if (degree <= 0) {
         throw std::runtime_error("degree must be lager than 0.");
      }
      
      RealType val = 0;
#pragma omp parallel for schedule(guided) reduction(+: val) num_threads(num_threads)
      for (std::int32_t i = 0; i < static_cast<std::int32_t>(samples.size()); ++i) {
         RealType avg = CalculateMagnetization(samples[i]);
         RealType prod = 1;
         for (std::int32_t j = 0; j < degree; ++j) {
            prod = prod*avg;
         }
         val += prod;
      }
      return val/samples.size();
   }
   
   RealType CalculateCorrelation(const std::vector<std::vector<OPType>> &samples,
                                 const IndexType ind1,
                                 const IndexType ind2) const {
      const std::unordered_map<IndexType, std::int32_t, IndexHash> &index_map = interaction_.GetIndexMap();
      if (index_map.count(ind1) == 0 || index_map.count(ind2) == 0) {
         throw std::runtime_error("The index is out of range.");
      }
      RealType val = 0;
      for (std::size_t i = 0; i < samples.size(); ++i) {
         val += samples[i][index_map.at(ind1)]*samples[i][index_map.at(ind2)];
      }
      return val/samples.size();
   }
   
private:
   interaction::PolynomialAnyInteraction<RealType> interaction_;
   lattice::AnyLattice lattice_;
   
   RealType CalculateMagnetization(const std::vector<OPType> &sample) const {
      RealType val = 0;
      for (std::size_t i = 0; i < sample.size(); ++i) {
         val += sample[i];
      }
      return val/sample.size();
   }
   
};

template<class LatticeType, typename RealType>
auto make_polynomial_ising(const LatticeType &lattice,
                           const typename PolynomialIsing<LatticeType, RealType>::PolynomialType &interaction) {
   return PolynomialIsing<LatticeType, RealType>{lattice, interaction};
}

} // namespace model
} // namespace compnal


#endif /* COMPNAL_MODEL_POLYNOMIAL_ISING_HPP_ */
