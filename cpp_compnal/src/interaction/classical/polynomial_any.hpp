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
//  polynomial_any.hpp
//  compnal
//
//  Created by kohei on 2022/08/13.
//  
//

#ifndef COMPNAL_INTERACTION_POLYNOMIAL_ANY_HPP_
#define COMPNAL_INTERACTION_POLYNOMIAL_ANY_HPP_

#include "../../utility/hash.hpp"
#include "../../utility/type.hpp"
#include "../../utility/sort.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace compnal {
namespace interaction {
namespace classical {

//! @brief Class to represent any polynomial-interactions.
//! @tparam RealType The value type, which must be floating point type.
template<typename RealType>
class PolynomialAny {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   //! @brief The index type.
   using IndexType = utility::AnyIndexType;
   
   //! @brief The hash for IndexType.
   using IndexHash = utility::AnyIndexHash;
   
   //! @brief The polynomial interaction type.
   using PolynomialType = std::unordered_map<std::vector<IndexType>, RealType, utility::AnyIndexVectorHash>;
   
   
   //! @brief Constructor for PolynomialAny class.
   //! @param interaction The polynomial interaction.
   PolynomialAny(const PolynomialType &interaction) {
      std::unordered_set<IndexType, IndexHash> index_set;
      for (const auto &it: interaction) {
         index_set.insert(it.first.begin(), it.first.end());
      }
      
      index_list_ = std::vector<IndexType>(index_set.begin(), index_set.end());
      std::sort(index_list_.begin(), index_list_.end());
      
      std::int32_t count = 0;
      for (std::size_t i = 0; i < index_list_.size(); ++i) {
         index_map_[index_list_[i]] = count;
         count++;
      }
      
      std::unordered_map<std::vector<std::int32_t>, RealType, utility::VectorHash> poly;
      poly.reserve(interaction.size());
      for (const auto &it: interaction) {
         std::vector<std::int32_t> keys(it.first.size());
         for (std::size_t i = 0; i < it.first.size(); ++i) {
            keys[i] = index_map_.at(it.first[i]);
         }
         std::sort(keys.begin(), keys.end());
         poly[keys] += it.second;
         if (degree_ < keys.size()) {
            degree_ = static_cast<std::int32_t>(keys.size());
         }
      }
      
      key_value_list_.reserve(poly.size());
      for (const auto &it: poly) {
         key_value_list_.push_back({it.first, it.second});
      }
      
      poly.clear();
      
      //Sort by keys.
      std::sort(key_value_list_.begin(), key_value_list_.end(), [](const auto &a, const auto &b) {
         return a.first < b.first;
      });
      
      adjacency_list_.resize(index_list_.size());
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         for (const auto &index: key_value_list_[i].first) {
            adjacency_list_[index].push_back(i);
         }
      }
      
      for (std::size_t i = 0; i < index_list_.size(); ++i) {
         // Save memory
         adjacency_list_[i].shrink_to_fit();
         std::sort(adjacency_list_[i].begin(), adjacency_list_[i].end());
      }
   }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return static_cast<std::int32_t>(index_list_.size());
   }
   
   //! @brief Get the degree of the polynomial interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      return degree_;
   }
   
   //! @brief Get the integer key and value list as pair.
   //! @return The integer key and value list as pair.
   const std::vector<std::pair<std::vector<std::int32_t>, RealType>> &GetKeyValueList() const {
      return key_value_list_;
   }
   
   //! @brief Get the index list of the polynomial interactions.
   //! @return The index list.
   const std::vector<IndexType> &GetIndexList() const {
      return index_list_;
   }
   
   //! @brief Get the mapping from the index to the integer.
   //! @return The index map.
   const std::unordered_map<IndexType, std::int32_t, IndexHash> &GetIndexMap() const {
      return index_map_;
   }
   
   //! @brief Get the adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   //! @return The adjacency list.
   const std::vector<std::vector<std::size_t>> &GetAdjacencyList() const {
      return adjacency_list_;
   }
   
private:
   //! @brief The degree of the polynomial interactions.
   int32_t degree_ = 0;
   
   //! @brief The mapping from the index to the integer.
   std::unordered_map<IndexType, std::int32_t, IndexHash> index_map_;
   
   //! @brief The integer key and value list as pair.
   std::vector<std::pair<std::vector<std::int32_t>, RealType>> key_value_list_;
   
   //! @brief The index list of the polynomial interactions.
   std::vector<IndexType> index_list_;
   
   //! @brief The adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   std::vector<std::vector<std::size_t>> adjacency_list_;
   
};

} // namespace classical
} // namespace interaction
} // namespace compnal


#endif /* COMPNAL_INTERACTION_POLYNOMIAL_ANY_HPP_ */
