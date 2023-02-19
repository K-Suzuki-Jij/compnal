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
//  Created by Kohei Suzuki on 2022/06/10.
//

#ifndef COMPNAL_MODEL_POLYNOMIAL_ISING_HPP_
#define COMPNAL_MODEL_POLYNOMIAL_ISING_HPP_

#include "../../lattice/all.hpp"
#include "../../interaction/classical/polynomial_any.hpp"
#include "../../utility/type.hpp"
#include <vector>
#include <unordered_map>

namespace compnal {
namespace model {
namespace classical {

//! @brief Class for representing PolynomialIsing class.
//! \f[ J_{0} + J_{1}\sum_{i}s_{i} + J_{2}\sum_{i,j}s_{i}s_{j} +
//! J_{3}\sum_{i,j,k}s_{i}s_{j}s_{k} + \ldots,\quad s_{i}\in\{-1,+1\} \f]
//! This class can represents the following models.
//! - PolynomialIsing model on the one-dimensional chain.
//! - PolynomialIsing model on the square lattice.
//! - PolynomialIsing model on the cubic lattice.
//! - Fully-connected PolynomialIsing model.
//!
//! @tparam LatticeType The lattice type, which must the following type.
//! - compnal::lattice::Chain
//! - compnal::lattice::Square
//! - compnal::lattice::Cubic
//! - compnal::lattice::InfiniteRange
//!
//! @tparam RealType The value type, which must be floating point type.
template<class LatticeType, typename RealType>
class PolynomialIsing {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief Coordinate index type.
   using IndexType = typename LatticeType::IndexType;
   
   //! @brief The operator type, which here represents Ising spins \f$ s_i\in \{-1,+1\} \f$
   using OPType = std::int8_t;
   
   //! @brief The type of polynomial interaction \f$ J_{p}\f$, here \f$ p \f$ is the degree of interactions.
   //! For example, \f$J_{2}=J_{3}=1.0\f$ can be set by {{2, 1.0}, {3, 1.0}} as std::unordered_map.
   using PolynomialType = std::unordered_map<std::int32_t, ValueType>;
   
   //! @brief Constructor for PolynomialIsing class.
   //! @param lattice The lattice.
   //! @param interaction The polynomial interaction.
   PolynomialIsing(const LatticeType &lattice,
                   const PolynomialType &interaction):
   lattice_(lattice) {
      for (const auto &it: interaction) {
         if (std::abs(it.second) <= std::numeric_limits<ValueType>::epsilon()) {
            continue;
         }
         if (interaction_.size() <= it.first) {
            interaction_.resize(it.first + 1);
         }
         interaction_[it.first] = it.second;
      }
   }
   
   //! @brief Get the system size \f$ N\f$.
   //! @return The system size \f$ N\f$.
   std::int32_t GetSystemSize() const {
      return lattice_.GetSystemSize();
   }
   
   //! @brief Get boundary condition.
   //! @return Boundary condition.
   lattice::BoundaryCondition GetBoundaryCondition() const {
      return lattice_.GetBoundaryCondition();
   }
   
   //! @brief Get polynomial interaction \f$ J_{p}\f$, here \f$ p \f$ is the degree of interactions.
   //! @return The polynomial interaction \f$ J_{p}\f$.
   //! For example, {0.0, 1.0, 1.0} as std::vector means \f$J_{2}=J_{3}=1.0\f$.
   const std::vector<ValueType> &GetInteraction() const {
      return interaction_;
   }
   
   //! @brief Get the lattice.
   //! @return The lattice.
   const LatticeType &GetLattice() const {
      return lattice_;
   }

   //! @brief Get the degree of the interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      return static_cast<std::int32_t>(interaction_.size()) - 1;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const std::vector<OPType> &spins) const {
      return CalculateEnergy(lattice_, spins);
   }
   
   
private:
   //! @brief The lattice.
   LatticeType lattice_;
   
   //! @brief The polynomial interaction.
   std::vector<ValueType> interaction_;
   
   //! @brief Calculate energy corresponding to the spin configuration on the one-dimensional chain.
   //! @param lattice The one-dimensional chain.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::Chain &lattice,
                            const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the two-dimensional square lattice.
   //! @param lattice The two-dimensional square lattice.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::Square &lattice,
                            const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the three-dimensional cubic lattice.
   //! @param lattice The three-dimensional cubic lattice.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::Cubic &lattice,
                            const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the infinite range lattice.
   //! @param lattice The infinite range lattice.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::InfiniteRange &lattice,
                            const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      return energy;
   }
   
};

//! @brief Class for representing the PolynomialIsing models on any lattices.
//! \f[ J^{(0)} + \sum_{i}J^{(1)}_{i}s_{i} + \sum_{i,j}J^{(2)}_{i,j}s_{i}s_{j} +
//! \sum_{i,j,k}J^{(3)}_{i,j,k}s_{i}s_{j}s_{k} + \ldots,\quad s_{i}\in\{-1,+1\} \f]
//! @tparam RealType The value type, which must be floating point type.
template<typename RealType>
class PolynomialIsing<lattice::AnyLattice, RealType> {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   //! @brief The value type.
   using ValueType = RealType
   ;
   //! @brief The index type.
   using IndexType = typename interaction::classical::PolynomialAny<ValueType>::IndexType;
   
   //! @brief The hash for IndexType.
   using IndexHash = typename interaction::classical::PolynomialAny<ValueType>::IndexHash;
   
   //! @brief The operator type, which here represents Ising spins \f$ s_i\in \{-1,+1\} \f$
   using OPType = std::int8_t;
   
   //! @brief The type of polynomial interaction \f$ J^{p}_{i,j,k,\ldots}\f$,
   //! here \f$ p \f$ is the degree of interactions.
   //! For example, \f$J^{2}_{1,2}=J^{3}_{1,2,3}=1.0\f$ can be set by
   //! {{{1, 2}, 1.0}, {{1, 2, 3}, 1.0}} as std::unordered_map.
   using PolynomialType = typename interaction::classical::PolynomialAny<ValueType>::PolynomialType;
   
   //! @brief Constructor for PolynomialIsing class.
   //! @param lattice The lattice.
   //! @param interaction The polynomial interaction.
   PolynomialIsing(const lattice::AnyLattice &lattice,
                   const PolynomialType &interaction):
   lattice_(lattice), interaction_(interaction) {}
   
   //! @brief Get the system size \f$ N\f$.
   //! @return The system size \f$ N\f$.
   std::int32_t GetSystemSize() const {
      return interaction_.GetSystemSize();
   }
   
   //! @brief Get boundary condition.
   //! @return Boundary condition.
   lattice::BoundaryCondition GetBoundaryCondition() const {
      return lattice_.GetBoundaryCondition();
   }
   
   //! @brief Get the integer key and value list as pair.
   //! @return The integer key and value list as pair.
   const std::vector<std::pair<std::vector<std::int32_t>, ValueType>> &GetKeyValueList() const {
      return interaction_.GetKeyValueList();
   }
   
   //! @brief Get the adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   //! @return The adjacency list.
   const std::vector<std::vector<std::size_t>> &GetAdjacencyList() const {
      return interaction_.GetAdjacencyList();
   }
   
   //! @brief Generate index list.
   //! @return The index list.
   const std::vector<IndexType> &GetIndexList() const {
      return interaction_.GetIndexList();
   }
   
   //! @brief Get the mapping from the index to the integer.
   //! @return The index map.
   const std::unordered_map<IndexType, std::int32_t, IndexHash> &GetIndexMap() const {
      return interaction_.GetIndexMap();
   }

   //! @brief Get the degree of the interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      return interaction_.GetDegree();
   }
   
   //! @brief Get the lattice.
   //! @return The lattice::AnyLattice object.
   const lattice::AnyLattice &GetLattice() const {
      return lattice_;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const std::vector<OPType> &spins) const {
      if (spins.size() != interaction_.GetSystemSize()) {
         throw std::runtime_error("The sample size is not equal to the system size");
      }
      const auto &key_list = interaction_.GetKeyList();
      const auto &value_list = interaction_.GetValueList();
      ValueType val = 0;
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         OPType spin = 1;
         for (const auto &index: key_list[i]) {
            spin *= spins[index];
         }
         val += spin*value_list[i];
      }
      return val;
   }
   
   
private:
   //! @brief The interaction.
   interaction::classical::PolynomialAny<ValueType> interaction_;
   
   //! @brief The linear interaction.
   lattice::AnyLattice lattice_;
      
};

//! @brief Helper function to make PolynomialIsing class.
//! @tparam LatticeType The lattice type.
//! @tparam RealType The value type, which must be floating point type.
//! @param lattice The lattice.
//! @param interaction The polynomial interaction.
template<class LatticeType, typename RealType>
auto make_polynomial_ising(const LatticeType &lattice,
                           const typename PolynomialIsing<LatticeType, RealType>::PolynomialType &interaction) {
   return PolynomialIsing<LatticeType, RealType>{lattice, interaction};
}

} // namespace classical
} // namespace model
} // namespace compnal


#endif /* COMPNAL_MODEL_POLYNOMIAL_ISING_HPP_ */
