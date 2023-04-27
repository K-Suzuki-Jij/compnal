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
//  square.hpp
//  compnal
//
//  Created by kohei on 2023/04/28.
//  
//

#ifndef COMPNAL_LATTICE_SQUARE_HPP_
#define COMPNAL_LATTICE_SQUARE_HPP_

#include <stdexcept>
#include <vector>

#include "boundary_condition.hpp"

namespace compnal {
namespace lattice {
   
class Square {
public:
   //! @brief Constructor.
   //! @param x_size The size of the x-direction.
   //! @param y_size The size of the y-direction.
   //! @param bc Boundary condtion. BoundaryCondition::NONE cannot be used here.
   Square(const std::int32_t x_size, const std::int32_t y_size,
          const BoundaryCondition bc) {
      if (x_size <= 0) {
         throw std::runtime_error("x_size must be larger than 0.");
      }
      if (y_size <= 0) {
         throw std::runtime_error("y_size must be larger than 0.");
      }
      if (bc == BoundaryCondition::NONE) {
         throw std::runtime_error("BoundaryCondition::NONE cannot be set.");
      }
      x_size_ = x_size;
      y_size_ = y_size;
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
   
   //! @brief Generate coordinate list in the square lattice.
   //! @return The coordinate list.
   std::vector<std::pair<std::int32_t, std::int32_t>> GenerateCoordinateList() const {
      std::vector<std::pair<std::int32_t, std::int32_t>> coo_list(x_size_*y_size_);
      for (std::int32_t i = 0; i < y_size_; ++i) {
         for (std::int32_t j = 0; j < x_size_; ++j) {
            coo_list[i*x_size_ + j] = {i, j};
         }
      }
      return coo_list;
   }
   
private:
   //! @brief Size of the x-direction.
   std::int32_t x_size_ = -1;
   
   //! @brief Size of the y-direction.
   std::int32_t y_size_ = -1;
   
   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::NONE;
   
};


}  // namespace lattice
}  // namespace compnal


#endif /* COMPNAL_LATTICE_SQUARE_HPP_ */
