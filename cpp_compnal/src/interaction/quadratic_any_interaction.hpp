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
//  quadratic_any_interaction.hpp
//  compnal
//
//  Created by kohei on 2022/08/13.
//  
//

#ifndef COMPNAL_INTERACTION_QUADRATIC_ANY_INTERACTION_HPP_
#define COMPNAL_INTERACTION_QUADRATIC_ANY_INTERACTION_HPP_

#include "../utility/hash.hpp"
#include "../utility/type.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace compnal {
namespace interaction {

//! @brief Class to represent any linear- and quadratic-interactions.
//! @tparam RealType The value type, which must be floating point type.
template<typename RealType>
class QuadraticAnyInteraction {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
public:
   //! @brief The index type.
   using IndexType = utility::AnyIndexType;
   
   //! @brief The hash for IndexType.
   using IndexHash = utility::AnyIndexHash;
   
   //! @brief The linear interaction type.
   using LinearType = std::unordered_map<IndexType, RealType, IndexHash>;
   
   //! @brief The quadratic interaction type.
   using QuadraticType = std::unordered_map<std::pair<IndexType, IndexType>, RealType, utility::AnyIndexPairHash>;
   
   //! @brief Constructor for QuadraticAnyInteraction class.
   //! @param linear The linear interaction.
   //! @param quadratic The quadratic interaction.
   QuadraticAnyInteraction(const LinearType &linear,
                           const QuadraticType &quadratic) {
      
      std::unordered_set<IndexType, IndexHash> index_set;
      for (const auto &it: linear) {
         if (std::abs(it.second) > std::numeric_limits<RealType>::epsilon()) {
            index_set.emplace(it.first);
         }
      }
      for (const auto &it: quadratic) {
         if (std::abs(it.second) > std::numeric_limits<RealType>::epsilon()) {
            index_set.emplace(it.first.first);
            index_set.emplace(it.first.second);
         }
      }
      
      index_list_ = std::vector<IndexType>(index_set.begin(), index_set.end());
      std::sort(index_list_.begin(), index_list_.end());
      
      std::int32_t count = 0;
      for (std::size_t i = 0; i < index_list_.size(); ++i) {
         index_map_[index_list_[i]] = count;
         count++;
      }
      
      linear_.resize(index_list_.size());
      for (const auto &it: linear) {
         if (std::abs(it.second) > std::numeric_limits<RealType>::epsilon()) {
            linear_[index_map_.at(it.first)] = it.second;
            degree_ = 1;
         }
      }
      
      std::unordered_map<std::pair<std::int32_t, std::int32_t>, RealType, utility::AnyIndexPairHash> new_quadratic;
      for (const auto &it: quadratic) {
         if (std::abs(it.second) > std::numeric_limits<RealType>::epsilon()) {
            const std::int32_t key_1 = index_map_.at(it.first.first);
            const std::int32_t key_2 = index_map_.at(it.first.second);
            if (key_1 < key_2) {
               new_quadratic[{key_1, key_2}] += it.second;
               degree_ = 2;
            }
            else if (key_1 > key_2) {
               new_quadratic[{key_2, key_1}] += it.second;
               degree_ = 2;
            }
            else {
               constant_ += it.second;
            }
         }
      }
      
      std::vector<std::vector<std::pair<std::int32_t, RealType>>> adjacency_list(index_list_.size());
      for (const auto &it: new_quadratic) {
         adjacency_list[it.first.first].push_back({it.first.second, it.second});
         adjacency_list[it.first.second].push_back({it.first.first, it.second});
      }
      
      for (std::size_t i = 0; i < adjacency_list.size(); ++i) {
         std::sort(adjacency_list[i].begin(), adjacency_list[i].end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
         });
      }
      
      row_ptr_.resize(adjacency_list.size() + 1);
      row_ptr_[0] = 0;
      for (std::size_t i = 0; i < adjacency_list.size(); ++i) {
         row_ptr_[i + 1] = static_cast<std::int64_t>(adjacency_list[i].size());
         for (const auto it: adjacency_list[i]) {
            col_ptr_.push_back(it.first);
            val_ptr_.push_back(it.second);
         }
      }
      
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(row_ptr_.size()) - 1; ++i) {
         row_ptr_[i + 1] += row_ptr_[i];
      }
      
   }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return static_cast<std::int32_t>(index_list_.size());
   }
   
   //! @brief Get the degree of the interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      return degree_;
   }
   
   //! @brief Get the index list of the interactions.
   //! @return The index list.
   const std::vector<IndexType> &GetIndexList() const {
      return index_list_;
   }
   
   //! @brief Get the mapping from the index to the integer.
   //! @return The index map.
   const std::unordered_map<IndexType, std::int32_t, IndexHash> &GetIndexMap() const {
      return index_map_;
   }
   
   //! @brief Get the constant value of the interactions,
   //! which appears when the interactions with the same index are set.
   //! @return The constant value.
   RealType GetConstant() const {
      return constant_;
   }
   
   //! @brief Get the linear interaction.
   //! @return The linear interaction.
   const std::vector<RealType> &GetLinear() const {
      return linear_;
   }
   
   //! @brief Get rows of the quadratic interaction as CRS format.
   //! @return The rows.
   const std::vector<std::int64_t> &GetRowPtr() const {
      return row_ptr_;
   }
   
   //! @brief Get columns of the quadratic interaction as CRS format.
   //! @return The columns.
   const std::vector<std::int32_t> &GetColPtr() const {
      return col_ptr_;
   }
   
   //! @brief Get values of the quadratic interaction as CRS format.
   //! @return The values.
   const std::vector<RealType> &GetValPtr() const {
      return val_ptr_;
   }
   
private:
   //! @brief The degree of the interactions.
   int32_t degree_ = 0;
   
   //! @brief The mapping from the index to the integer.
   std::unordered_map<IndexType, std::int32_t, IndexHash> index_map_;
   
   //! @brief The the constant value of the interactions,
   //! which appears when the interactions with the same index are set.
   RealType constant_ = 0;
   
   //! @brief The linear interaction.
   std::vector<RealType> linear_;
   
   //! @brief The index list of the interactions.
   std::vector<IndexType> index_list_;
   
   //! @brief The rows of the quadratic interaction as CRS format.
   std::vector<std::int64_t> row_ptr_;
   
   //! @brief The columns of the quadratic interaction as CRS format.
   std::vector<std::int32_t> col_ptr_;
   
   //! @brief The values of the quadratic interaction as CRS format.
   std::vector<RealType> val_ptr_;
   
};


} // namespace interaction
} // namespace compnal

#endif /* COMPNAL_INTERACTION_QUADRATIC_ANY_INTERACTION_HPP_ */
