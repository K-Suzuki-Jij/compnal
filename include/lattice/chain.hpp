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
//  chain.hpp
//  compnal
//
//  Created by kohei on 2023/04/24.
//
//

#ifndef COMPNAL_LATTICE_CHAIN_HPP_
#define COMPNAL_LATTICE_CHAIN_HPP_

#include <numeric>
#include <stdexcept>

#include "boundary_condition.hpp"

namespace compnal {
namespace lattice {

//! @brief The class to represent the one-dimensional chain.
class Chain {
public:
   //! @brief Constructor.
   //! @param system_size System size.
   //! @param bc Boundary condtion. BoundaryCondition::NONE cannot be used here.
   Chain(const std::int32_t system_size, const BoundaryCondition bc) {
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
   std::int32_t GetSystemSize() const { return system_size_; }

   //! @brief Get boundary condition.
   //! @return Boundary condition.
   BoundaryCondition GetBoundaryCondition() const { return bc_; }

   //! @brief Generate coordinate list in the chain.
   //! @return The coordinate list.
   std::vector<std::int32_t> GenerateCoordinateList() const {
      std::vector<std::int32_t> index_list(system_size_);
      std::iota(index_list.begin(), index_list.end(), 0);
      return index_list;
   }

private:
   //! @brief System size.
   std::int32_t system_size_ = 0;

   //! @brief Boundary condition.
   BoundaryCondition bc_ = BoundaryCondition::NONE;
};

}  // namespace lattice
}  // namespace compnal

#endif /* COMPNAL_LATTICE_CHAIN_HPP_ */
