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
//  polynomial_ising.hpp
//  compnal
//
//  Created by kohei on 2023/05/03.
//  
//

#ifndef COMPNAL_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_
#define COMPNAL_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_

#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "../../lattice/all.hpp"
#include "../utility/variable.hpp"
#include "../../utility/combination.hpp"

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
template<class LatticeType>
class PolynomialIsing {
   
public:
   //! @brief The physical quantity type, which represents Ising spins \f$ s_i\in \{-S, -S+1, \ldots, S\} \f$.
   //! To represent half-integers, double is set here.
   using PHQType = double;
   
   //! @brief Constructor for PolynomialIsing class.
   //! @param lattice The lattice.
   //! @param interaction The polynomial interaction.
   //! @param spin_magnitude The magnitude of spins. This must be half-integer.
   PolynomialIsing(const LatticeType &lattice,
                   const std::unordered_map<std::int32_t, double> &interaction,
                   const double spin_magnitude = 0.5,
                   const std::int32_t spin_scale_factor = 1):
   lattice_(lattice) {
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
      
      for (const auto &it: interaction) {
         if (it.first < 0) {
            throw std::invalid_argument("The degree of interactions must be positive.");
         }
         if (std::abs(it.second) <= std::numeric_limits<double>::epsilon()) {
            continue;
         }
         if (degree_ < it.first) {
            degree_ = it.first;
         }
         interaction_[it.first] = it.second;
      }
   }
   
   const std::unordered_map<std::int32_t, double> &GetInteraction() const {
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
      return degree_;
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
   LatticeType lattice_;
   
   //! @brief The polynomial interaction.
   std::unordered_map<std::int32_t, double> interaction_;
   
   //! @brief The degree of the interactions.
   std::int32_t degree_ = 0;
   
   //! @brief Twice magnitude of spins.
   std::vector<std::int32_t> twice_spin_magnitude_;
   
   //! @brief spin_scale_factor A scaling factor used to adjust the value taken by the spin.
   //! The default value is 1.0, which represents the usual spin, taking value s\in\{-1/2,+1/2\}.
   //! By changing this value, you can represent spins of different values,
   //! such as s\in\{-1,+1\} by setting spin_scale_factor=2.
   std::int32_t spin_scale_factor_ = 1;
   
   //! @brief Calculate energy corresponding to the spin configuration on the one-dimensional chain.
   //! @param lattice The one-dimensional chain.
   //! @param state The spin configuration.
   //! @return The energy.
   template<class VecType>
   double CalculateEnergy(const lattice::Chain &lattice,
                          const VecType &state) const {
      double energy = 0;
      const std::int32_t system_size = lattice.GetSystemSize();
      
      if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
         for (const auto &it: interaction_) {
            if (it.first == 0) {
               energy += it.second;
            }
            else {
               for (std::int32_t index = 0; index < system_size; ++index) {
                  PHQType spin_prod = state[index];
                  for (std::int32_t p = 1; p < it.first; ++p) {
                     spin_prod *= state[(index + p)%system_size];
                  }
                  energy += it.second*spin_prod;
               }
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            if (it.first == 0) {
               energy += it.second;
            }
            else {
               for (std::int32_t index = 0; index < system_size; ++index) {
                  if (index + it.first - 1 >= system_size) {
                     break;
                  }
                  PHQType spin_prod = state[index];
                  for (std::int32_t p = 1; p < it.first; ++p) {
                     spin_prod *= state[index + p];
                  }
                  energy += it.second*spin_prod;
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Unsupported BoundaryCondition");
      }
      
      return energy;
   }
   
   //! @brief Calculate energy corresponding to the spin configuration on the two-dimensional square lattice.
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
         for (const auto &it: interaction_) {
            if (it.first == 0) {
               energy += it.second;
            }
            else if (it.first == 1) {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     energy += it.second*state[coo_y*x_size + coo_x];
                  }
               }
            }
            else {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     const std::int32_t index = coo_y*x_size + coo_x;
                     PHQType spin_prod_x = state[index];
                     PHQType spin_prod_y = state[index];
                     for (std::int32_t p = 1; p < it.first; ++p) {
                        spin_prod_x *= state[coo_y*x_size + (coo_x + p)%x_size];
                        spin_prod_y *= state[((coo_y + p)%y_size)*x_size + coo_x];
                     }
                     energy += it.second*(spin_prod_x + spin_prod_y);
                  }
               }
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            if (it.first == 0) {
               energy += it.second;
            }
            else if (it.first == 1) {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     energy += it.second*state[coo_y*x_size + coo_x];
                  }
               }
            }
            else {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     const std::int32_t index = coo_y*x_size + coo_x;
                     if (coo_x + it.first - 1 < x_size) {
                        PHQType spin_prod_x = state[index];
                        for (std::int32_t p = 1; p < it.first; ++p) {
                           spin_prod_x *= state[coo_y*x_size + coo_x + p];
                        }
                        energy += it.second*spin_prod_x;
                     }
                     if (coo_y + it.first - 1 < y_size) {
                        PHQType spin_prod_y = state[index];
                        for (std::int32_t p = 1; p < it.first; ++p) {
                           spin_prod_y *= state[(coo_y + p)*x_size + coo_x];
                        }
                        energy += it.second*spin_prod_y;
                     }
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
   
   //! @brief Calculate energy corresponding to the spin configuration on the three-dimensional cubic lattice.
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
         for (const auto &it: interaction_) {
            if (it.first == 0) {
               energy += it.second;
            }
            else if (it.first == 1) {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                        energy += it.second*state[coo_z*x_size*y_size + coo_y*x_size + coo_x];
                     }
                  }
               }
            }
            else {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                        const std::int32_t index = coo_y*x_size + coo_x;
                        PHQType spin_prod_x = state[index];
                        PHQType spin_prod_y = state[index];
                        PHQType spin_prod_z = state[index];
                        for (std::int32_t p = 1; p < it.first; ++p) {
                           spin_prod_x *= state[coo_z*x_size*y_size + coo_y*x_size + (coo_x + p)%x_size];
                           spin_prod_y *= state[coo_z*x_size*y_size + ((coo_y + p)%y_size)*x_size + coo_x];
                           spin_prod_z *= state[((coo_z + p)%z_size)*x_size*y_size + coo_y*x_size + coo_x];
                        }
                        energy += it.second*(spin_prod_x + spin_prod_y + spin_prod_z);
                     }
                  }
               }
            }
         }
      }
      else if (lattice.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (const auto &it: interaction_) {
            if (it.first == 0) {
               energy += it.second;
            }
            else if (it.first == 1) {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                        energy += it.second*state[coo_z*x_size*y_size + coo_y*x_size + coo_x];
                     }
                  }
               }
            }
            else {
               for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
                  for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
                     for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                        const std::int32_t index = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                        if (coo_x + it.first - 1 < x_size) {
                           PHQType spin_prod_x = state[index];
                           for (std::int32_t p = 1; p < it.first; ++p) {
                              spin_prod_x *= state[coo_z*x_size*y_size + coo_y*x_size + coo_x + p];
                           }
                           energy += it.second*spin_prod_x;
                        }
                        if (coo_y + it.first - 1 < y_size) {
                           PHQType spin_prod_y = state[index];
                           for (std::int32_t p = 1; p < it.first; ++p) {
                              spin_prod_y *= state[coo_z*x_size*y_size + (coo_y + p)*x_size + coo_x];
                           }
                           energy += it.second*spin_prod_y;
                        }
                        if (coo_z + it.first - 1 < z_size) {
                           PHQType spin_prod_z = state[index];
                           for (std::int32_t p = 1; p < it.first; ++p) {
                              spin_prod_z *= state[(coo_z + p)*x_size*y_size + coo_y*x_size + coo_x];
                           }
                           energy += it.second*spin_prod_z;
                        }
                     }
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
   //! @param lattice The infinite range lattice.
   //! @param state The spin configuration.
   //! @return The energy.
   template<class VecType>
   double CalculateEnergy(const lattice::InfiniteRange &lattice,
                          const VecType &state) const {
      double energy = 0;
      const std::int32_t system_size = lattice.GetSystemSize();
      
      for (const auto &it: interaction_) {
         if (it.first == 0) {
            energy += it.second;
            continue;
         }
         std::vector<std::int32_t> indices(it.first);
         std::int32_t start_index = 0;
         std::int32_t size = 0;
         
         while (true) {
            for (std::int32_t i = start_index; i < system_size; ++i) {
               indices[size++] = i;
               if (size == it.first) {
                  PHQType sign = 1;
                  for (std::int32_t j = 0; j < it.first; ++j) {
                     sign *= state[indices[j]];
                  }
                  energy += it.second*sign;
                  break;
               }
            }
            --size;
            if (size < 0) {
               break;
            }
            start_index = indices[size] + 1;
         }
      }
      return energy;
   }
   
};

//! @brief Helper function to make PolynomialIsing class.
//! @tparam LatticeType The lattice type.
//! @param lattice The lattice.
//! @param interaction The polynomial interaction.
//! @param spin_magnitude The magnitude of spins. This must be half-integer.
//! @param spin_scale_factor A scaling factor used to adjust the value taken by the spin.
//! The default value is 1.0, which represents the usual spin, taking value s\in\{-1/2,+1/2\}.
//! By changing this value, you can represent spins of different values,
//! such as s\in\{-1,+1\} by setting spin_scale_factor=2.
template<class LatticeType>
auto make_polynomial_ising(const LatticeType &lattice,
                           const std::unordered_map<std::int32_t, double> &interaction,
                           const double spin_magnitude = 0.5,
                           const std::int32_t spin_scale_factor = 1) {
   return PolynomialIsing<LatticeType>{lattice, interaction, spin_magnitude, spin_scale_factor};
}

} // namespace classical
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_ */
