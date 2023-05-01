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
//  Created by kohei on 2023/05/01.
//  
//

#ifndef COMPNAL_MODEL_CLASSICAL_ISING_HPP_
#define COMPNAL_MODEL_CLASSICAL_ISING_HPP_

namespace compnal {
namespace model {
namespace classical {

//! @brief Class for representing classical Ising model.
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
template<class LatticeType>
class Ising {
   
public:
   //! @brief The physical quantity type, which represents Ising spins \f$ s_i\in \{-1,+1\} \f$
   using PHQType = std::int8_t;
   
   //! @brief Constructor for Ising class.
   //! @param lattice The lattice.
   //! @param linear The linear interaction.
   //! @param quadratic The quadratic interaction.
   Ising(const LatticeType &lattice,
         const double linear,
         const double quadratic):
   lattice_(lattice), linear_(linear), quadratic_(quadratic) {}
   
   //! @brief Get linear interaction \f$ h\f$.
   //! @return The linear interaction \f$ h\f$.
   double GetLinear() const {
      return linear_;
   }
   
   //! @brief Get quadratic interaction \f$ J\f$.
   //! @return The quadratic interaction \f$ J\f$.
   double GetQuadratic() const {
      return quadratic_;
   }
   
   //! @brief Get the lattice.
   //! @return The lattice.
   const LatticeType &GetLattice() const {
      return lattice_;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration.
   //! @param spins The spin configuration.
   //! @return The energy.
   double CalculateEnergy(const std::vector<PHQType> &spins) const {
      if (spins.size() != lattice_.GetSystemSize()) {
         throw std::runtime_error("The system size is not equal to the size of spins");
      }
      return CalculateEnergy(lattice_, spins);
   }
   
private:
   //! @brief The lattice.
   const LatticeType lattice_;
   
   //! @brief The linear interaction.
   const double linear_ = 0.0;
   
   //! @brief The quadratic interaction.
   const double quadratic_ = 0.0;
   
   //! @brief Calculate energy corresponding to the spin configuration on the one-dimensional chain.
   //! @param lattice The one-dimensional chain.
   //! @param spins The spin configuration.
   //! @return The energy.
   double CalculateEnergy(const lattice::Chain &lattice,
                          const std::vector<PHQType> &spins) const {
      double energy = 0;
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
   double CalculateEnergy(const lattice::Square &lattice,
                          const std::vector<PHQType> &spins) const {
      double energy = 0;
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
   double CalculateEnergy(const lattice::Cubic &lattice,
                          const std::vector<PHQType> &spins) const {
      
      double energy = 0;
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
   double CalculateEnergy(const lattice::InfiniteRange &lattice,
                          const std::vector<PHQType> &spins) const {
      double energy = 0;
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



} // namespace classical
} // namespace model
} // namespace compnal


#endif /* COMPNAL_MODEL_CLASSICAL_ISING_HPP_ */
