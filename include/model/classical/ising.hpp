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

#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include "../../lattice/all.hpp"
#include "../utility/variable.hpp"

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
   //! @brief The physical quantity type, which represents Ising spins \f$ s_i\in \{-S, -S+1, \ldots, S\} \f$.
   //! To represent half-integers, double is set here.
   using PHQType = double;
   
   //! @brief Constructor for Ising class.
   //! @param lattice The lattice.
   //! @param linear The linear interaction.
   //! @param quadratic The quadratic interaction.
   //! @param spin_magnitude The magnitude of spins. This must be half-integer.
   //! @param spin_scale_factor A scaling factor used to adjust the value taken by the spin.
   //! The default value is 1.0, which represents the usual spin, taking value \f[ s\in\{-1/2,+1/2\} \f].
   //! By changing this value, you can represent spins of different values,
   //! such as \f[ s\in\{-1,+1\} \f] by setting spin_scale_factor=2.
   Ising(const LatticeType &lattice,
         const double linear,
         const double quadratic,
         const double spin_magnitude = 0.5,
         const std::int32_t spin_scale_factor = 1):
   lattice_(lattice), linear_(linear), quadratic_(quadratic) {
      if (std::floor(2*spin_magnitude) != 2*spin_magnitude || spin_magnitude <= 0) {
         throw std::invalid_argument("spin_magnitude must be positive half-integer.");
      }
      if (spin_scale_factor < 1) {
         throw std::invalid_argument("spin_scale_factor must positive-integer");
      }
      spin_scale_factor_ = spin_scale_factor;
      twice_spin_magnitude_.resize(lattice.GetSystemSize());
      for (std::int32_t i = 0; i < lattice.GetSystemSize(); ++i) {
         twice_spin_magnitude_[i] = static_cast<std::int32_t>(2*spin_magnitude);
      }
   }
   
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
   
   //! @brief Get twice the magnitude of the spins.
   //! @return Twice the magnitude of the spins.
   const std::vector<std::int32_t> &GetTwiceSpinMagnitude() const {
      return twice_spin_magnitude_;
   }
   
   //! @brief Get spin-scale factor.
   //! @return Spin-scale factor.
   std::int32_t GetSpinScaleFactor() const {
      return spin_scale_factor_;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration.
   //! @param state The spin configuration.
   //! @return The energy.
   double CalculateEnergy(const std::vector<PHQType> &state) const {
      if (state.size() != lattice_.GetSystemSize()) {
         throw std::range_error("The system size is not equal to the size of spins");
      }
      return CalculateEnergy(lattice_, state);
   }
   
   //! @brief Calculate energy corresponding to the spin configuration.
   //! @param state The spin configuration.
   //! @return The energy.
   double CalculateEnergy(const Eigen::Vector<PHQType, Eigen::Dynamic> &state) const {
      if (state.size() != lattice_.GetSystemSize()) {
         throw std::range_error("The system size is not equal to the size of spins");
      }
      return CalculateEnergy(lattice_, state);
   }
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const std::vector<model::utility::Spin> &sample) const {
      if (sample.size() != lattice_.GetSystemSize()) {
         throw std::range_error("The system size is not equal to the size of spins");
      }
      return GenerateEnergyDifference(lattice_, sample);
   }
   
   //! @brief Set the magnitude of the spin.
   //! @param spin_magnitude The magnitude of the spin. This must be half-integer.
   //! @param coordinate The coordinate.
   void SetSpinMagnitude(const double spin_magnitude, const typename LatticeType::CoordinateType coordinate) {
      if (std::floor(2*spin_magnitude) != 2*spin_magnitude) {
         throw std::invalid_argument("magnitude must be half-integer.");
      }
      if (!lattice_.ValidateCoordinate(coordinate)) {
         throw std::invalid_argument("The input coordinate is not in the system");
      }
      twice_spin_magnitude_[lattice_.CoordinateToInteger(coordinate)] = static_cast<std::int32_t>(2*spin_magnitude);
   }
   
private:
   //! @brief The lattice.
   const LatticeType lattice_;
   
   //! @brief The linear interaction.
   const double linear_ = 0.0;
   
   //! @brief The quadratic interaction.
   const double quadratic_ = 0.0;
   
   //! @brief Twice magnitude of spins.
   std::vector<std::int32_t> twice_spin_magnitude_;
   
   //! @brief spin_scale_factor A scaling factor used to adjust the value taken by the spin.
   //! The default value is 1.0, which represents the usual spin, taking value \f] s\in\{-1/2,+1/2\} \f].
   //! By changing this value, you can represent spins of different values,
   //! such as \f[ s\in\{-1,+1\} \f] by setting spin_scale_factor=2.
   std::int32_t spin_scale_factor_ = 1;
   
   //! @brief Calculate energy corresponding to the spin configuration on the one-dimensional chain.
   //! @tparam VecType The vector type.
   //! @param lattice The one-dimensional chain.
   //! @param state The spin configuration.
   //! @return The energy.
   template<class VecType>
   double CalculateEnergy(const lattice::Chain &lattice,
                          const VecType &state) const {
      double energy = 0;
      const std::int32_t system_size = lattice.GetSystemSize();
      
      for (std::int32_t index = 0; index < system_size - 1; ++index) {
         energy += quadratic_*state[index]*state[index + 1] + linear_*state[index];
      }
      
      if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
         energy += quadratic_*state[system_size - 1]*state[0] + linear_*state[system_size - 1];
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         energy += linear_*state[system_size - 1];
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the two-dimensional square lattice.
   //! @tparam VecType The vector type.
   //! @param lattice The two-dimensional square lattice.
   //! @param state The spin configuration.
   //! @return The energy.
   template<class VecType>
   double CalculateEnergy(const lattice::Square &lattice,
                          const VecType &state) const {
      double energy = 0;
      const std::int32_t x_size = lattice.GetXSize();
      const std::int32_t y_size = lattice.GetYSize();
      
      if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t index    = coo_y*x_size + coo_x;
               const std::int32_t index_x1 = coo_y*x_size + (coo_x + 1)%x_size;
               const std::int32_t index_y1 = ((coo_y + 1)%y_size)*x_size + coo_x;
               energy += linear_*state[index];
               energy += quadratic_*state[index]*state[index_x1];
               energy += quadratic_*state[index]*state[index_y1];
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t index = coo_y*x_size + coo_x;
               energy += linear_*state[index];
               if (coo_x < x_size - 1) {
                  const std::int32_t index_x1 = coo_y*x_size + (coo_x + 1);
                  energy += quadratic_*state[index]*state[index_x1];
               }
               if (coo_y < y_size - 1) {
                  const std::int32_t index_y1 = (coo_y + 1)*x_size + coo_x;
                  energy += quadratic_*state[index]*state[index_y1];
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the three-dimensional cubic lattice.
   //! @tparam VecType The vector type.
   //! @param lattice The three-dimensional cubic lattice.
   //! @param state The spin configuration.
   //! @return The energy.
   template<class VecType>
   double CalculateEnergy(const lattice::Cubic &lattice,
                          const VecType &state) const {
      
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
                  energy += linear_*state[index];
                  energy += quadratic_*state[index]*state[index_x1];
                  energy += quadratic_*state[index]*state[index_y1];
                  energy += quadratic_*state[index]*state[index_z1];
               }
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t index = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  energy += linear_*state[index];
                  if (coo_x < x_size - 1) {
                     const std::int32_t index_x1 = coo_z*x_size*y_size + coo_y*x_size + (coo_x + 1);
                     energy += quadratic_*state[index]*state[index_x1];
                  }
                  if (coo_y < y_size - 1) {
                     const std::int32_t index_y1 = coo_z*x_size*y_size + (coo_y + 1)*x_size + coo_x;
                     energy += quadratic_*state[index]*state[index_y1];
                  }
                  if (coo_z < z_size - 1) {
                     const std::int32_t index_z1 = (coo_z + 1)*x_size*y_size + coo_y*x_size + coo_x;
                     energy += quadratic_*state[index]*state[index_z1];
                  }
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the infinite range lattice.
   //! @tparam VecType The vector type.
   //! @param lattice The infinite range lattice.
   //! @param state The spin configuration.
   //! @return The energy.
   template<class VecType>
   double CalculateEnergy(const lattice::InfiniteRange &lattice,
                          const VecType &state) const {
      double energy = 0;
      const std::int32_t system_size = lattice.GetSystemSize();
      for (std::int32_t i = 0; i < system_size; ++i) {
         energy += linear_*state[i];
         for (std::int32_t j = i + 1; j < system_size; ++j) {
            energy += quadratic_*state[i]*state[j];
         }
      }
      return energy;
   }
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const lattice::Chain &lattice, 
                                                const std::vector<model::utility::Spin> &sample) const {
      const std::int32_t system_size = lattice.GetSystemSize();
      const auto bc = lattice.GetBoundaryCondition();
      std::vector<double> d_E_(system_size);
      if (bc == lattice::BoundaryCondition::PBC) {
         for (std::int32_t index = 0; index < system_size; ++index) {
            const auto v1 = sample[(index - 1 + system_size)%system_size].GetValue();
            const auto v2 = sample[(index + 1)%system_size].GetValue();
            d_E_[index] += this->quadratic_*(v1 + v2) + this->linear_;
         }
      }
      else if (bc == lattice::BoundaryCondition::OBC) {
         for (std::int32_t index = 0; index < system_size; ++index) {
            if (index < system_size - 1) {
               d_E_[index] += this->quadratic_*sample[index + 1].GetValue();
            }
            if (index > 0) {
               d_E_[index] += this->quadratic_*sample[index - 1].GetValue();
            }
            d_E_[index] += this->linear_;
         }
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      return d_E_;
   }
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const lattice::Square &lattice,
                                                const std::vector<model::utility::Spin> &sample) const {
      const std::int32_t system_size = lattice.GetSystemSize();
      const std::int32_t x_size = lattice.GetXSize();
      const std::int32_t y_size = lattice.GetYSize();
      const auto bc = lattice.GetBoundaryCondition();
      std::vector<double> d_E_(system_size);
      if (bc == lattice::BoundaryCondition::PBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t index_xp1 = (coo_y*x_size + (coo_x + 1)%x_size);
               const std::int32_t index_xm1 = (coo_y*x_size + (coo_x - 1 + x_size)%x_size);
               const std::int32_t index_yp1 = (((coo_y + 1)%y_size)*x_size + coo_x);
               const std::int32_t index_ym1 = (((coo_y - 1 + y_size)%y_size)*x_size + coo_x);
               const auto v_xp1 = sample[index_xp1].GetValue();
               const auto v_xm1 = sample[index_xm1].GetValue();
               const auto v_yp1 = sample[index_yp1].GetValue();
               const auto v_ym1 = sample[index_ym1].GetValue();
               d_E_[coo_y*x_size + coo_x] += this->quadratic_*(v_xp1 + v_xm1 + v_yp1 + v_ym1) + this->linear_;
            }
         }
      }
      else if (bc == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               if (coo_x < x_size - 1) {
                  d_E_[coo_y*x_size + coo_x] += this->quadratic_*sample[coo_y*x_size + coo_x + 1].GetValue();
               }
               if (coo_x > 0) {
                  d_E_[coo_y*x_size + coo_x] += this->quadratic_*sample[coo_y*x_size + coo_x - 1].GetValue();
               }
               if (coo_y < y_size - 1) {
                  d_E_[coo_y*x_size + coo_x] += this->quadratic_*sample[(coo_y + 1)*x_size + coo_x].GetValue();
               }
               if (coo_y > 0) {
                  d_E_[coo_y*x_size + coo_x] += this->quadratic_*sample[(coo_y - 1)*x_size + coo_x].GetValue();
               }
               d_E_[coo_y*x_size + coo_x] += this->linear_;
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      return d_E_;
   }
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const lattice::Cubic &lattice,
                                                const std::vector<model::utility::Spin> &sample) const {
      const std::int32_t system_size = lattice.GetSystemSize();
      const std::int32_t x_size = lattice.GetXSize();
      const std::int32_t y_size = lattice.GetYSize();
      const std::int32_t z_size = lattice.GetZSize();
      const auto bc = lattice.GetBoundaryCondition();
      std::vector<double> d_E_(system_size);
      if (bc == lattice::BoundaryCondition::PBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t index_xp1 = coo_z*x_size*y_size + coo_y*x_size + (coo_x + 1)%x_size;
                  const std::int32_t index_xm1 = coo_z*x_size*y_size + coo_y*x_size + (coo_x - 1 + x_size)%x_size;
                  const std::int32_t index_yp1 = coo_z*x_size*y_size + ((coo_y + 1)%y_size)*x_size + coo_x;
                  const std::int32_t index_ym1 = coo_z*x_size*y_size + ((coo_y - 1 + y_size)%y_size)*x_size + coo_x;
                  const std::int32_t index_zp1 = ((coo_z + 1)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  const std::int32_t index_zm1 = ((coo_z - 1 + z_size)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  
                  const auto v_xp1 = sample[index_xp1].GetValue();
                  const auto v_xm1 = sample[index_xm1].GetValue();
                  const auto v_yp1 = sample[index_yp1].GetValue();
                  const auto v_ym1 = sample[index_ym1].GetValue();
                  const auto v_zp1 = sample[index_zp1].GetValue();
                  const auto v_zm1 = sample[index_zm1].GetValue();
                  d_E_[coo_z*x_size*y_size + coo_y*x_size + coo_x] += this->quadratic_*(v_xp1 + v_xm1 + v_yp1 + v_ym1 + v_zp1 + v_zm1) + this->linear_;
               }
            }
         }
      }
      else if (bc == lattice::BoundaryCondition::OBC) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t index = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  if (coo_x < x_size - 1) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size*y_size + coo_y*x_size + coo_x + 1].GetValue();
                  }
                  if (coo_x > 0) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size*y_size + coo_y*x_size + coo_x - 1].GetValue();
                  }
                  if (coo_y < y_size - 1) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size*y_size + (coo_y + 1)*x_size + coo_x].GetValue();
                  }
                  if (coo_y > 0) {
                     d_E_[index] += this->quadratic_*sample[coo_z*x_size*y_size + (coo_y - 1)*x_size + coo_x].GetValue();
                  }
                  if (coo_z < z_size - 1) {
                     d_E_[index] += this->quadratic_*sample[(coo_z + 1)*x_size*y_size + coo_y*x_size + coo_x].GetValue();
                  }
                  if (coo_z > 0) {
                     d_E_[index] += this->quadratic_*sample[(coo_z - 1)*x_size*y_size + coo_y*x_size + coo_x].GetValue();
                  }
                  d_E_[index] += this->linear_;
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BinaryCondition");
      }
      return d_E_;
      
   }
   
   //! @brief Generate energy difference.
   //! @param sample The spin configuration.
   //! @return The energy difference.
   std::vector<double> GenerateEnergyDifference(const lattice::InfiniteRange &lattice,
                                                const std::vector<model::utility::Spin> &sample) const {
      const std::int32_t system_size = lattice.GetSystemSize();
      std::vector<double> d_E_(system_size);
      for (std::int32_t i = 0; i < system_size; ++i) {
         d_E_[i] += this->linear_;
         for (std::int32_t j = i + 1; j < system_size; ++j) {
            d_E_[i] += this->quadratic_*sample[j].GetValue();
            d_E_[j] += this->quadratic_*sample[i].GetValue();
         }
      }
      return d_E_;
   }
   
};

//! @brief Helper function to make Ising class.
//! @tparam LatticeType The lattice type.
//! @param lattice The lattice.
//! @param linear The linear interaction.
//! @param quadratic The quadratic interaction.
//! @param spin_magnitude The magnitude of spins. This must be half-integer.
//! @param spin_scale_factor A scaling factor used to adjust the value taken by the spin.
//! The default value is 1.0, which represents the usual spin, taking value \f[ s\in\{-1/2,+1/2\} \f].
//! By changing this value, you can represent spins of different values,
//! such as \f[ s\in\{-1,+1\} \f] by setting spin_scale_factor=2.
template<class LatticeType>
auto make_ising(const LatticeType &lattice,
                const double linear,
                const double quadratic,
                const double spin_magnitude = 0.5,
                const std::int32_t spin_scale_factor = 1) {
   return Ising<LatticeType>{lattice, linear, quadratic, spin_magnitude, spin_scale_factor};
}

} // namespace classical
} // namespace model
} // namespace compnal
