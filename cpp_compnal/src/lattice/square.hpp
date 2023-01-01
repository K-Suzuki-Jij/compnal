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
//  square.hpp
//  compnal
//
//  Created by kohei on 2022/08/10.
//  
//

#ifndef COMPNAL_LATTICE_SQUARE_HPP_
#define COMPNAL_LATTICE_SQUARE_HPP_

#include "base_two_dimensional_lattice.hpp"
#include <vector>

namespace compnal {
namespace lattice {

//! @brief Class to represent the two-dimensional square lattice.
class Square: public BaseTwoDimensionalLattice {
public:
   using BaseTwoDimensionalLattice::BaseTwoDimensionalLattice;
   
   //! @brief Generate index list.
   //! @return The index list.
   std::vector<std::pair<std::int32_t, std::int32_t>> GenerateIndexList() const {
      std::vector<std::pair<std::int32_t, std::int32_t>> index_list(this->GetSystemSize());
      for (std::int32_t i = 0; i < this->GetYSize(); ++i) {
         for (std::int32_t j = 0; j < this->GetXSize(); ++j) {
            index_list[i*this->GetXSize() + j] = {i, j};
         }
      }
      return index_list;
   }
   
};

} // namespace lattice
} // namespace compnal

#endif /* COMPNAL_LATTICE_SQUARE_HPP_ */
