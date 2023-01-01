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
//  chain.hpp
//  compnal
//
//  Created by kohei on 2022/08/09.
//  
//

#ifndef COMPNAL_LATTICE_CHAIN_HPP_
#define COMPNAL_LATTICE_CHAIN_HPP_

#include "base_one_dimensional_lattice.hpp"
#include <vector>
#include <numeric>

namespace compnal {
namespace lattice {

//! @brief The class to represent the one-dimensional chain.
class Chain: public BaseOneDimensionalLattice {
public:
   using BaseOneDimensionalLattice::BaseOneDimensionalLattice;
   
   //! @brief Generate index list.
   //! @return The index list.
   std::vector<std::int32_t> GenerateIndexList() const {
      std::vector<std::int32_t> index_list(this->GetSystemSize());
      std::iota(index_list.begin(), index_list.end(), 0);
      return index_list;
   }
   
};

} // namespace lattice
} // namespace compnal

#endif /* COMPNAL_LATTICE_CHAIN_HPP_ */
