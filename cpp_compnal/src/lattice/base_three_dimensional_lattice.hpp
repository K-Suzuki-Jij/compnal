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
//  base_three_dimensional_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_LATTICE_BASE_THREE_DIMENSIONAL_LATTICE_HPP_
#define COMPNAL_LATTICE_BASE_THREE_DIMENSIONAL_LATTICE_HPP_

#include "boundary_condition.hpp"
#include <stdexcept>
#include <cstdint>
#include <utility>
#include <tuple>

namespace compnal {
namespace lattice {

//! @brief Base class to represent the three-dimensional lattice.
class BaseThreeDimensionalLattice {
  
public:
   //! @brief Coordinate index type.
   using IndexType = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
      
   //! @brief Constructor of BaseThreeDimensionalLattice.
   //! @param x_size The size of the x-direction.
   //! @param y_size The size of the y-direction.
   //! @param z_size The size of the z-direction.
   //! @param bc Boundary condtion. BoundaryCondition::NONE cannot be used here.
   BaseThreeDimensionalLattice(const std::int32_t x_size,
                               const std::int32_t y_size,
                               const std::int32_t z_size,
                               const BoundaryCondition bc = BoundaryCondition::OBC) {
      if (x_size <= 0) {
         throw std::runtime_error("x_size must be larger than 0.");
      }
      if (y_size <= 0) {
         throw std::runtime_error("y_size must be larger than 0.");
      }
      if (z_size <= 0) {
         throw std::runtime_error("z_size must be larger than 0.");
      }
      if (bc == BoundaryCondition::NONE) {
         throw std::runtime_error("BoundaryCondition::NONE cannot be set.");
      }
      x_size_ = x_size;
      y_size_ = y_size;
      z_size_ = z_size;
      bc_ = bc;
   }
      
   //! @brief Get size of the x-direction.
   //! @return Size of the x-direction.
   std::int32_t GetXSize() const {
      return x_size_;
   }
   
   //! @brief Get size of the y-direction.
   //! @return Size of the y-direction.
   std::int32_t GetYSize() const {
      return y_size_;
   }
   
   //! @brief Get size of the z-direction.
   //! @return Size of the z-direction.
   std::int32_t GetZSize() const {
      return z_size_;
   }
   
   //! @brief Get system size.
   //! @return System size.
   std::int32_t GetSystemSize() const {
      return x_size_*y_size_*z_size_;
   }
   
   //! @brief Get boundary condition.
   //! @return Boundary condition.
   BoundaryCondition GetBoundaryCondition() const {
      return bc_;
   }
   
   //! @brief Check if the value of the coordinate is in the system.
   //! @param site_index Value of the coordinate.
   //! @return True or False.
   bool ValidateCOOIndex(const IndexType site_index) const {
      if (std::get<0>(site_index)  >= x_size_ ||
          std::get<0>(site_index)  <  0       ||
          std::get<1>(site_index) >= y_size_  ||
          std::get<1>(site_index) <  0        ||
          std::get<2>(site_index) >= z_size_  ||
          std::get<2>(site_index) <  0       ) {
         return false;
      }
      else {
         return true;
      }
   }
   
   //! @brief Calculate site index as integer from the value of the coordinate.
   //! @param site_index Value of the coordinate.
   //! @return Site index as integer.
   std::int32_t CalculateIntegerSiteIndex(const IndexType site_index) const {
      return std::get<2>(site_index)*x_size_*y_size_ + std::get<1>(site_index)*x_size_ + std::get<0>(site_index);
   }
   
   
private:
   //! @brief Size of the x-direction.
   std::int32_t x_size_ = -1;
   
   //! @brief Size of the y-direction.
   std::int32_t y_size_ = -1;
   
   //! @brief Size of the z-direction.
   std::int32_t z_size_ = -1;
   
   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::OBC;
   
};



} // namespace lattice
} // namespace compnal


#endif /* COMPNAL_LATTICE_BASE_THREE_DIMENSIONAL_LATTICE_HPP_ */
