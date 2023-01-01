//
//  Copyright 2022 Kohei Suzuki
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
   
   //! @brief Constructor of BaseOneDimensionalLattice class.
   //! @param system_size System size.
   BaseOneDimensionalLattice(const std::int32_t system_size) {
      if (system_size < 0) {
         throw std::runtime_error("system_size must be larger than or equal to 0.");
      }
      system_size_ = system_size;
   }
   
   //! @brief Constructor of BaseOneDimensionalLattice class.
   //! @param system_size System size.
   //! @param boundary_condition Boundary condtion. BoundaryCondition::NONE cannot be used here.
   BaseOneDimensionalLattice(const std::int32_t system_size,
                             const BoundaryCondition boundary_condition): BaseOneDimensionalLattice(system_size) {
      if (boundary_condition == BoundaryCondition::NONE) {
         throw std::runtime_error("BoundaryCondition::NONE cannot be set.");
      }
      bc_ = boundary_condition;
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
   
private:
   //! @brief System size.
   std::int32_t system_size_ = 0;
   
   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::OBC;
   
};


} // namespace lattice
} // namespace compnal

#endif /* COMPNAL_LATTICE_BASE_ONE_DIMENSIONAL_LATTICE_HPP_ */
