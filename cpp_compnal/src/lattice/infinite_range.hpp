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
//  infinite_range.hpp
//  compnal
//
//  Created by kohei on 2022/08/10.
//  
//

#ifndef COMPNAL_LATTICE_INFINITE_RANGE_HPP_
#define COMPNAL_LATTICE_INFINITE_RANGE_HPP_

#include "boundary_condition.hpp"
#include <stdexcept>
#include <vector>
#include <numeric>

namespace compnal {
namespace lattice {

//! @brief Class to represent the infinite-dimensional lattice.
class InfiniteRange {
   
public:
   //! @brief Coordinate index type.
   using IndexType = std::int32_t;
   
   //! @brief Constructor of InfiniteRange class.
   //! @param system_size System size.
   InfiniteRange(const std::int32_t system_size) {
      if (system_size < 0) {
         throw std::runtime_error("system_size must be larger than or equal to 0.");
      }
      system_size_ = system_size;
   }
      
   //! @brief Get system size.
   //! @return System size.
   std::int32_t GetSystemSize() const {
      return system_size_;
   }
   
   //! @brief Get boundary condition.
   //! @return Boundary condition, BoundaryCondition::NONE.
   BoundaryCondition GetBoundaryCondition() const {
      return BoundaryCondition::NONE;
   }
   
   //! @brief Check if the value of the coordinate is in the system.
   //! @param site_index Value of the coordinate.
   //! @return True or False.
   bool ValidateCOOIndex(const IndexType site_index) const {
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
   std::int32_t CalculateIntegerSiteIndex(const IndexType site_index) const {
      return site_index;
   }
   
   //! @brief Generate index list.
   //! @return The index list.
   std::vector<std::int32_t> GenerateIndexList() const {
      std::vector<std::int32_t> index_list(system_size_);
      std::iota(index_list.begin(), index_list.end(), 0);
      return index_list;
   }
   
private:
   //! @brief System size.
   std::int32_t system_size_ = 0;
   
};

} // namespace lattice
} // namespace compnal


#endif /* COMPNAL_LATTICE_INFINITE_RANGE_HPP_ */
