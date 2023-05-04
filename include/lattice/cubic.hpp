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
//  cubic.hpp
//  compnal
//
//  Created by kohei on 2023/04/28.
//  
//

#ifndef COMPNAL_LATTICE_CUBIC_HPP_
#define COMPNAL_LATTICE_CUBIC_HPP_

#include <stdexcept>
#include <vector>
#include <tuple>

#include "boundary_condition.hpp"

namespace compnal {
namespace lattice {

//! @brief Class to represent the cubic lattice.
class Cubic {
   
public:
   //! @brief Coordinate type.
   using CoordinateType = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
   
   //! @brief Constructor.
   //! @param x_size The size of the x-direction.
   //! @param y_size The size of the y-direction.
   //! @param z_size The size of the z-direction.
   //! @param bc Boundary condtion. BoundaryCondition::NONE cannot be used here.
   Cubic(const std::int32_t x_size, const std::int32_t y_size, const std::int32_t z_size,
         const BoundaryCondition bc = BoundaryCondition::OBC) {
      if (x_size <= 0) {
         throw std::invalid_argument("x_size must be larger than 0.");
      }
      if (y_size <= 0) {
         throw std::invalid_argument("y_size must be larger than 0.");
      }
      if (z_size <= 0) {
         throw std::invalid_argument("z_size must be larger than 0.");
      }
      if (bc == BoundaryCondition::NONE) {
         throw std::invalid_argument("BoundaryCondition::NONE cannot be set.");
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
   
   //! @brief Generate coordinate list in the square lattice.
   //! @return The coordinate list.
   std::vector<CoordinateType> GenerateCoordinateList() const {
      std::vector<CoordinateType> coo_list(x_size_*y_size_*z_size_);
      for (std::int32_t i = 0; i < z_size_; ++i) {
         for (std::int32_t j = 0; j < y_size_; ++j) {
            for (std::int32_t k = 0; k < x_size_; ++k) {
               coo_list[i*y_size_*x_size_ + j*x_size_ + k] = {i, j, k};
            }
         }
      }
      return coo_list;
   }
   
   //! @brief Change input coordinate to an integer.
   //! @param coordinate The coordinate.
   //! @return Corresponding integer.
   std::int32_t CoordinateToInteger(const CoordinateType coordinate) const {
      return std::get<0>(coordinate)*y_size_*x_size_ + std::get<1>(coordinate)*x_size_ + std::get<2>(coordinate);
   }
   
   //! @brief Check if input coordinate is in the system.
   //! @param coordinate The coordinate.
   //! @return True if the coordinate is in the system, otherwise false.
   bool ValidateCoordinate(const CoordinateType coordinate) const {
      if ((0 <= std::get<2>(coordinate)) && (std::get<2>(coordinate) < x_size_) &&
          (0 <= std::get<1>(coordinate)) && (std::get<1>(coordinate) < y_size_) &&
          (0 <= std::get<0>(coordinate)) && (std::get<0>(coordinate) < z_size_)) {
         return true;
      }
      else {
         return false;
      }
   }
   
private:
   //! @brief Size of the x-direction.
   std::int32_t x_size_ = 0;
   
   //! @brief Size of the y-direction.
   std::int32_t y_size_ = 0;
   
   //! @brief Size of the z-direction.
   std::int32_t z_size_ = 0;
   
   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::NONE;
   
};

}  // namespace lattice
}  // namespace compnal

#endif /* COMPNAL_LATTICE_CUBIC_HPP_ */
