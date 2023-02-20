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
//  ising.hpp
//  compnal
//
//  Created by kohei on 2022/08/13.
//  
//

#ifndef COMPNAL_MODEL_ISING_HPP_
#define COMPNAL_MODEL_ISING_HPP_

#include "../../lattice/all.hpp"
#include "../../utility/type.hpp"
#include "../../interaction/classical/quadratic_any.hpp"
#include <vector>

namespace compnal {
namespace model {
namespace classical {

//! @brief Class for representing typical Ising models.
//! \f[ J\sum_{i,j}s_{i}s_{j} + h\sum_{i}s_{i},\quad s_{i}\in\{-1,+1\} \f]
//! This class can represents the following models.
//! - Ising model on the one-dimensional chain.
//! - Ising model on the square lattice.
//! - Ising model on the cubic lattice.
//! - Fully-connected Ising model.
//!
//! @tparam LatticeType The lattice type, which must the following type.
//! - compnal::lattice::Chain
//! - compnal::lattice::Square
//! - compnal::lattice::Cubic
//! - compnal::lattice::InfiniteRange
//!
//! @tparam RealType The value type, which must be floating point type.
template<class LatticeType, typename RealType>
class Ising {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief Coordinate index type.
   using IndexType = typename LatticeType::IndexType;
   
   //! @brief The operator type, which here represents Ising spins \f$ s_i\in \{-1,+1\} \f$
   using OPType = std::int8_t;
   
   //! @brief The type of linear interaction \f$ h\f$.
   using LinearType = ValueType;
   
   //! @brief The type of quadratic interaction \f$ J\f$.
   using QuadraticType = ValueType;
   
   //! @brief Constructor for Ising class.
   //! @param lattice The lattice.
   //! @param linear The linear interaction.
   //! @param quadratic The quadratic interaction.
   Ising(const LatticeType &lattice,
         const LinearType linear,
         const QuadraticType quadratic):
   lattice_(lattice), linear_(linear), quadratic_(quadratic) {}
   
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
   
   //! @brief Get linear interaction \f$ h\f$.
   //! @return The linear interaction \f$ h\f$.
   LinearType GetLinear() const {
      return linear_;
   }
   
   //! @brief Get quadratic interaction \f$ J\f$.
   //! @return The quadratic interaction \f$ J\f$.
   QuadraticType GetQuadratic() const {
      return quadratic_;
   }
   
   //! @brief Get the lattice.
   //! @return The lattice.
   const LatticeType &GetLattice() const {
      return lattice_;
   }
      
   //! @brief Get the degree of the interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      if (std::abs(quadratic_) > std::numeric_limits<ValueType>::epsilon()) {
         return 2;
      }
      else if (std::abs(linear_) > std::numeric_limits<ValueType>::epsilon()) {
         return 1;
      }
      else {
         return 0;
      }
   }
   
   //! @brief Calculate energy corresponding to the spin configuration.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const std::vector<OPType> &spins) const {
      if (spins.size() != lattice_.GetSystemSize()) {
         throw std::runtime_error("The system size is not equal to the size of spins");
      }
      return CalculateEnergy(lattice_, spins);
   }
   
private:
   //! @brief The lattice.
   const LatticeType lattice_;
   
   //! @brief The linear interaction.
   const LinearType linear_ = 0;
   
   //! @brief The quadratic interaction.
   const QuadraticType quadratic_ = 0;
   
   //! @brief Calculate energy corresponding to the spin configuration on the one-dimensional chain.
   //! @param lattice The one-dimensional chain.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::Chain &lattice,
                             const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      const std::int32_t system_size = lattice.GetSystemSize();
      
      for (std::int32_t index = 0; index < system_size - 1; ++index) {
         energy += quadratic_*spins[index]*spins[index + 1] + linear_*spins[index];
      }
      
      if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
         energy += quadratic_*spins[system_size - 1]*spins[0] + linear_*spins[system_size - 1];
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         energy += linear_*spins[system_size - 1];
      }
      else {
         throw std::runtime_error("Unsupported BinaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the two-dimensional square lattice.
   //! @param lattice The two-dimensional square lattice.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::Square &lattice,
                             const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      const std::int32_t x_size = lattice.GetXSize();
      const std::int32_t y_size = lattice.GetYSize();
      
      if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t index    = coo_y*x_size + coo_x;
               const std::int32_t index_x1 = coo_y*x_size + (coo_x + 1)%x_size;
               const std::int32_t index_y1 = ((coo_y + 1)%y_size)*x_size + coo_x;
               energy += linear_*spins[index];
               energy += quadratic_*spins[index]*spins[index_x1];
               energy += quadratic_*spins[index]*spins[index_y1];
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t index = coo_y*x_size + coo_x;
               energy += linear_*spins[index];
               if (coo_x < x_size - 1) {
                  const std::int32_t index_x1 = coo_y*x_size + (coo_x + 1);
                  energy += quadratic_*spins[index]*spins[index_x1];
               }
               if (coo_y < y_size - 1) {
                  const std::int32_t index_y1 = (coo_y + 1)*x_size + coo_x;
                  energy += quadratic_*spins[index]*spins[index_y1];
               }
            }
         }
      }
      else {
         throw std::runtime_error("Unsupported BoundaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the three-dimensional cubic lattice.
   //! @param lattice The three-dimensional cubic lattice.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::Cubic &lattice,
                             const std::vector<OPType> &spins) const {
      
      ValueType energy = 0;
      const std::int32_t x_size = lattice.GetXSize();
      const std::int32_t y_size = lattice.GetYSize();
      const std::int32_t z_size = lattice.GetZSize();
      
      if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t index    = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  const std::int32_t index_x1 = coo_z*x_size*y_size + coo_y*x_size + (coo_x + 1)%x_size;
                  const std::int32_t index_y1 = coo_z*x_size*y_size + ((coo_y + 1)%y_size)*x_size + coo_x;
                  const std::int32_t index_z1 = ((coo_z + 1)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  energy += linear_*spins[index];
                  energy += quadratic_*spins[index]*spins[index_x1];
                  energy += quadratic_*spins[index]*spins[index_y1];
                  energy += quadratic_*spins[index]*spins[index_z1];
               }
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t index = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  energy += linear_*spins[index];
                  if (coo_x < x_size - 1) {
                     const std::int32_t index_x1 = coo_z*x_size*y_size + coo_y*x_size + (coo_x + 1);
                     energy += quadratic_*spins[index]*spins[index_x1];
                  }
                  if (coo_y < y_size - 1) {
                     const std::int32_t index_y1 = coo_z*x_size*y_size + (coo_y + 1)*x_size + coo_x;
                     energy += quadratic_*spins[index]*spins[index_y1];
                  }
                  if (coo_z < z_size - 1) {
                     const std::int32_t index_z1 = (coo_z + 1)*x_size*y_size + coo_y*x_size + coo_x;
                     energy += quadratic_*spins[index]*spins[index_z1];
                  }
               }
            }
         }
      }
      else {
         throw std::runtime_error("Unsupported BoundaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the infinite range lattice.
   //! @param lattice The infinite range lattice.
   //! @param spins The spin configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const lattice::InfiniteRange &lattice,
                             const std::vector<OPType> &spins) const {
      ValueType energy = 0;
      const std::int32_t system_size = lattice.GetSystemSize();
      for (std::int32_t i = 0; i < system_size; ++i) {
         energy += linear_*spins[i];
         for (std::int32_t j = i + 1; j < system_size; ++j) {
            energy += quadratic_*spins[i]*spins[j];
         }
      }
      return energy;
   }
   
};

//! @brief Class for representing the classical Ising models on any lattices.
//! \f[ \sum_{i,j}J_{i,j}s_{i}s_{j} + \sum_{i}h_{i}s_{i},\quad s_{i}\in\{-1,+1\} \f]
//! @tparam RealType The value type, which must be floating point type.
template<typename RealType>
class Ising<lattice::AnyLattice, RealType> {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief The index type.
   using IndexType = typename interaction::classical::QuadraticAny<ValueType>::IndexType;
   
   //! @brief The hash for IndexType.
   using IndexHash = typename interaction::classical::QuadraticAny<ValueType>::IndexHash;
   
   //! @brief The operator type, which here represents Ising spins \f$ s_i\in \{-1,+1\} \f$
   using OPType = std::int8_t;
   
   //! @brief The linear interaction type.
   using LinearType = typename interaction::classical::QuadraticAny<ValueType>::LinearType;
   
   //! @brief The quadratic interaction type.
   using QuadraticType = typename interaction::classical::QuadraticAny<ValueType>::QuadraticType;
   
   //! @brief Constructor for Ising class.
   //! @param lattice The lattice, which must be compnal::lattice::AnyLattice.
   //! @param linear The linear interaction.
   //! @param quadratic The quadratic interaction.
   Ising(const lattice::AnyLattice &lattice,
         const LinearType &linear,
         const QuadraticType &quadratic):
   lattice_(lattice), interaction_(linear, quadratic) {}
   
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
   
   //! @brief Get the constant value of the interactions.
   //! @return The constant value.
   ValueType GetConstant() const {
      return interaction_.GetConstant();
   }
   
   //! @brief Get linear interaction.
   //! @return The linear interaction.
   const std::vector<ValueType> &GetLinear() const {
      return interaction_.GetLinear();
   }
   
   //! @brief Get rows of the quadratic interaction as CRS format.
   //! @return The rows.
   const std::vector<std::int64_t> &GetRowPtr() const {
      return interaction_.GetRowPtr();
   }
   
   //! @brief Get columns of the quadratic interaction as CRS format.
   //! @return The columns.
   const std::vector<std::int32_t> &GetColPtr() const {
      return interaction_.GetColPtr();
   }
   
   //! @brief Get values of the quadratic interaction as CRS format.
   //! @return The values.
   const std::vector<ValueType> &GetValPtr() const {
      return interaction_.GetValPtr();
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
         throw std::runtime_error("The size of spin configuration is not equal to the system size.");
      }
      ValueType val = interaction_.GetConstant();
      const std::int32_t system_size = interaction_.GetSystemSize();
      const auto &linear = GetLinear();
      const auto &row_ptr = GetRowPtr();
      const auto &col_ptr = GetColPtr();
      const auto &val_ptr = GetValPtr();
      for (std::int32_t i = 0; i < system_size; ++i) {
         for (std::int64_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (i < col_ptr[j]) {
               val += val_ptr[j]*spins[col_ptr[j]]*spins[i];
            }
         }
         val += linear[i]*spins[i];
      }
      return val;
   }
   
private:
   //! @brief The interaction.
   const interaction::classical::QuadraticAny<ValueType> interaction_;
   
   //! @brief The linear interaction.
   const lattice::AnyLattice lattice_;
   
};

//! @brief Helper function to make Ising class.
//! @tparam LatticeType The lattice type.
//! @tparam RealType The value type, which must be floating point type.
//! @param lattice The lattice.
//! @param linear The linear interaction.
//! @param quadratic The quadratic interaction.
template<class LatticeType, typename RealType>
auto make_ising(const LatticeType &lattice,
                const typename Ising<LatticeType, RealType>::LinearType &linear,
                const typename Ising<LatticeType, RealType>::QuadraticType &quadratic) {
   return Ising<LatticeType, RealType>{lattice, linear, quadratic};
}

} // namespace classical
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_ISING_HPP_ */
