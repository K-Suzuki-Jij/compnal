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
//  base_one_dimensional_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_LATTICE_BASE_ONE_DIMENSIONAL_LATTICE_HPP_
#define COMPNAL_LATTICE_BASE_ONE_DIMENSIONAL_LATTICE_HPP_

#include "boundary_condition.hpp"
#include <stdexcept>
#include <cstdint>

namespace compnal {
namespace lattice {

//! @brief Base class to represent the one-dimensional system.
class BaseOneDimensionalLattice {
   
public:
   //! @brief Cordinate index type.
   using COOIndexType = std::int32_t;
      
   //! @brief Constructor of BaseOneDimensionalLattice class.
   //! @param system_size System size.
   //! @param bc Boundary condtion. BoundaryCondition::NONE cannot be used here.
   //! Defaults to BoundaryCondition::OBC.
   BaseOneDimensionalLattice(const std::int32_t system_size,
                             const BoundaryCondition bc = BoundaryCondition::OBC) {
      if (system_size <= 0) {
         throw std::runtime_error("system_size must be larger than 0.");
      }
      if (bc == BoundaryCondition::NONE) {
         throw std::runtime_error("BoundaryCondition::NONE cannot be set.");
      }
      system_size_ = system_size;
      bc_ = bc;
   }
      
   //! @brief Get system size.
   //! @return System size.
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
   //! @brief Get boundary condition.
   //! @return Boundary condition.
   BoundaryCondition GetBoundaryCondition() const {
      return bc_;
   }
   
   //! @brief Check if the value of the coordinate is in the system.
   //! @param site_index Value of the coordinate.
   //! @return True or False.
   bool ValidateCOOIndex(const COOIndexType site_index) const {
      if (site_index  >= system_size_ || site_index < 0) {
         return false;
      }
      else {
         return true;
      }
   }
   
   //! @brief Calculate site index as integer from the value of the coordinate.
   //! This function return the same value of the input.
   //! @param site_index Value of the coordinate.
   //! @return Site index as integer.
   std::int32_t CalculateIntegerSiteIndex(const COOIndexType site_index) const {
      return site_index;
   }
   
private:
   //! @brief System size.
   std::int32_t system_size_ = -1;
   
   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::OBC;
   
};


} // namespace lattice
} // namespace compnal

#endif /* COMPNAL_LATTICE_BASE_ONE_DIMENSIONAL_LATTICE_HPP_ */
