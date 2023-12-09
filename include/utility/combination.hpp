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
//  combination.hpp
//  compnal
//
//  Created by kohei on 2023/06/29.
//  
//

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

namespace compnal {
namespace utility {

//! @brief Get the next combination.
//! @tparam Iterator Iterator type.
//! @param first Iterator to the first element.
//! @param k Specify the number of selection.
//! @param last Iterator to the last element.
//! @return If the next combination exists, return true.
template<typename Iterator>
bool NextCombination(const Iterator first, Iterator k, const Iterator last) {
   if (first == last || first == k || last == k) {
      return false;
   }
   
   Iterator itr1 = first;
   Iterator itr2 = last;
   itr1++;
   
   if (last == itr1) {
      return false;
   }
   
   itr1 = k;
   itr2--;
   
   while (first != itr1) {
      if (*--itr1 < *itr2) {
         Iterator j = k;
         
         while (*itr1 >= *j) {
            j++;
         };
         
         std::iter_swap(itr1, j);
         itr1++;
         j++;
         itr2 = k;
         std::rotate(itr1, j, last);
         
         while (last != j) {
            j++;
            itr2++;
         }
         
         std::rotate(k, itr2, last);
         return true;
      }
   }
   std::rotate(first, k, last);
   return false;
}

} // namespace utility
} // namespace compnal
