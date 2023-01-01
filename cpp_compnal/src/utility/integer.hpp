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
//  Created by Kohei Suzuki on 2022/07/06.
//

#ifndef COMPNAL_UTILITY_INTEGER_HPP_
#define COMPNAL_UTILITY_INTEGER_HPP_

#include <vector>

namespace compnal {
namespace utility {

//! @brief Generate all the combinations from a given list.
//! @param list The list, from which all the combinations will be generated.
//! @param combination_size The size of combinations.
//! @return All the combinations.
template<typename T>
std::vector<std::vector<T>> GenerateAllCombinations(const std::vector<T> &list, const std::int64_t combination_size) {
   
   std::vector<std::vector<T>> combinations;
   
   if (static_cast<std::int64_t>(list.size()) < combination_size) {
      throw std::runtime_error("combination_size must be smallar than or equal to list.size().");
   }
   if (combination_size < 0) {
      throw std::runtime_error("combination_size must be larger than -1.");
   }
   if (combination_size == 0) {
      return combinations;
   }
   
   std::vector<std::int64_t> indices(combination_size);
   
   const std::int64_t seed_size = static_cast<std::int64_t>(list.size());
   std::int64_t start_index = 0;
   std::int64_t size = 0;
   
   while (true) {
      for (std::int64_t i = start_index; i < seed_size; ++i) {
         indices[size++] = i;
         if (size == combination_size) {
            std::vector<T> temp(combination_size);
            for (std::size_t j = 0; j < temp.size(); ++j) {
               temp[j] = list[indices[j]];
            }
            combinations.push_back(temp);
            break;
         }
      }
      --size;
      if (size < 0) {
         break;
      }
      start_index = indices[size] + 1;
   }
   combinations.shrink_to_fit();
   
   return combinations;
}

} // namespace utility
} // namespace compnal


#endif /* COMPNAL_UTILITY_INTEGER_HPP_ */
