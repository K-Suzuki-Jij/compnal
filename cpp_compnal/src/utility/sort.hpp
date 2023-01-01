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
//  sort.hpp
//  compnal
//
//  Created by kohei on 2022/10/13.
//  
//

#ifndef COMPNAL_UTILITY_SORT_HPP_
#define COMPNAL_UTILITY_SORT_HPP_

namespace compnal {
namespace utility {

// Ascending order
template <typename T1, typename T2>
void QuickSortVector(std::vector<T1> *target_vector, std::vector<T2> *vector_1, const std::int64_t left, const std::int64_t right) {
   if (right - left <= 1) {
      return;
   }

   const std::int64_t pivot_index = (left + right) / 2;
   const T1 pivot = (*target_vector)[pivot_index];

   std::swap((*target_vector)[pivot_index], (*target_vector)[right - 1]);
   std::swap((*vector_1)[pivot_index], (*vector_1)[right - 1]);

   std::int64_t index = left;
   for (std::int64_t i = left; i < right - 1; i++) {
      if ((*target_vector)[i] < pivot) {
         std::swap((*target_vector)[index], (*target_vector)[i]);
         std::swap((*vector_1)[index], (*vector_1)[i]);
         index++;
      }
   }

   std::swap((*target_vector)[index], (*target_vector)[right - 1]);
   std::swap((*vector_1)[index], (*vector_1)[right - 1]);

   QuickSortVector(target_vector, vector_1, left, index);
   QuickSortVector(target_vector, vector_1, index + 1, right);
}

// Ascending order
template <typename T1, typename T2>
void QuickSortVectorBySize(std::vector<T1> *target_vector, std::vector<T2> *vector_1, const std::int64_t left, const std::int64_t right) {
   if (right - left <= 1) {
      return;
   }

   const std::int64_t pivot_index = (left + right) / 2;
   const std::size_t pivot = (*target_vector)[pivot_index].size();

   std::swap((*target_vector)[pivot_index], (*target_vector)[right - 1]);
   std::swap((*vector_1)[pivot_index], (*vector_1)[right - 1]);

   std::int64_t index = left;
   for (std::int64_t i = left; i < right - 1; i++) {
      if ((*target_vector)[i].size() < pivot) {
         std::swap((*target_vector)[index], (*target_vector)[i]);
         std::swap((*vector_1)[index], (*vector_1)[i]);
         index++;
      }
   }

   std::swap((*target_vector)[index], (*target_vector)[right - 1]);
   std::swap((*vector_1)[index], (*vector_1)[right - 1]);

   QuickSortVectorBySize(target_vector, vector_1, left, index);
   QuickSortVectorBySize(target_vector, vector_1, index + 1, right);
}

}  // namespace utility
}  // namespace compnal


#endif /* COMPNAL_UTILITY_SORT_HPP_ */
