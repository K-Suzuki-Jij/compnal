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
//  base_two_dimensional_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_LATTICE_BASE_TWO_DIMENSIONAL_LATTICE_HPP_
#define COMPNAL_LATTICE_BASE_TWO_DIMENSIONAL_LATTICE_HPP_

#include "boundary_condition.hpp"
#include <stdexcept>
#include <cstdint>

namespace compnal {
namespace lattice {

//! @brief Base class to represent the two-dimensional lattice.
class BaseTwoDimensionalLattice {
   
public:
   using COOIndexType = std::pair<std::int32_t, std::int32_t>;
   
   //! @brief Constructor of BaseTwoDimensionalLattice class.
   //! @param x_size The size of the x-direction.
   //! @param y_size The size of the y-direction.
   BaseTwoDimensionalLattice(const std::int32_t x_size,
                             const std::int32_t y_size) {
      if (x_size <= 0) {
         throw std::runtime_error("x_size must be larger than 0.");
      }
      if (y_size <= 0) {
         throw std::runtime_error("y_size must be larger than 0.");
      }
      
      x_size_ = x_size;
      y_size_ = y_size;
   }
   
   //! @brief Constructor of BaseTwoDimensionalLattice class.
   //! @param x_size The size of the x-direction.
   //! @param y_size The size of the y-direction.
   //! @param boundary_condition Boundary condtion. BoundaryCondition::NONE cannot be used here.
   BaseTwoDimensionalLattice(const std::int32_t x_size,
                             const std::int32_t y_size,
                             const BoundaryCondition boundary_condition): BaseTwoDimensionalLattice(x_size, y_size) {
      if (boundary_condition == BoundaryCondition::NONE) {
         throw std::runtime_error("BoundaryCondition::NONE cannot be set.");
      }
      bc_ = boundary_condition;
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
   
   //! @brief Get system size.
   //! @return System size.
   std::int32_t GetSystemSize() const {
      return x_size_*y_size_;
   }
   
   //! @brief Get boundary condition.
   //! @return Boundary condition.
   BoundaryCondition GetBoundaryCondition() const {
      return bc_;
   }
   
   bool ValidateCOOIndex(const COOIndexType site_index) const {
      if (site_index.first  >= x_size_ ||
          site_index.first  <  0       ||
          site_index.second >= y_size_ ||
          site_index.second <  0       ) {
         return false;
      }
      else {
         return true;
      }
   }
   
   std::int32_t CalculateOneDimSiteIndex(const COOIndexType site_index) const {
      return site_index.second*x_size_ + site_index.first;
   }
   
   
private:
   //! @brief Size of the x-direction.
   std::int32_t x_size_ = -1;
   
   //! @brief Size of the y-direction.
   std::int32_t y_size_ = -1;
   
   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::OBC;
   
};


} // namespace lattice
} // namespace compnal

#endif /* COMPNAL_LATTICE_BASE_TWO_DIMENSIONAL_LATTICE_HPP_ */
