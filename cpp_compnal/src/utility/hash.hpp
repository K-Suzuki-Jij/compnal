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
//  hash.hpp
//  compnal
//
//  Created by kohei on 2022/08/10.
//  
//

#ifndef COMPNAL_UTILITY_HASH_HPP_
#define COMPNAL_UTILITY_HASH_HPP_

#include <variant>
#include "type.hpp"

namespace compnal {
namespace utility {

//! @brief Hash struct of AnyIndexType
struct AnyIndexHash {
   template<class... Types>
   std::size_t operator()(const std::variant<Types...> &v) const {
      if (std::holds_alternative<std::int32_t>(v)) {
         return std::hash<std::int32_t>()(std::get<std::int32_t>(v));
      }
      else if (std::holds_alternative<std::string>(v)) {
         return std::hash<std::string>()(std::get<std::string>(v));
      }
      else if (std::holds_alternative<AnyTupleType>(v)) {
         const auto &variant_vec = std::get<AnyTupleType>(v);
         std::size_t hash = variant_vec.size();
         for (const auto &i : variant_vec) {
            if (std::holds_alternative<std::int32_t>(i)) {
               hash ^= std::hash<std::int32_t>()(std::get<std::int32_t>(i)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            else if (std::holds_alternative<std::string>(i)) {
               hash ^= std::hash<std::string>()(std::get<std::string>(i)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            else {
               throw std::runtime_error("Invalid template parameters");
            }
         }
         return hash;
      }
      else {
         throw std::runtime_error("Invalid template parameters");
      }
   }
};

//! @brief Hash struct of std::vector<AnyIndexType>.
struct AnyIndexVectorHash {
   std::size_t operator()(const std::vector<AnyIndexType> &v) const {
      std::size_t hash = v.size();
      for (const auto &i : v) {
         hash ^= AnyIndexHash()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
   }
};

//! @brief Hash struct of std::pair<AnyIndexType>.
struct AnyIndexPairHash {
   std::size_t operator()(const std::pair<AnyIndexType, AnyIndexType> &p) const {
      std::size_t lhs = AnyIndexHash()(p.first);
      std::size_t rhs = AnyIndexHash()(p.second);
      return lhs^(rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2));
   }
};

} // namespace utility
} // namespace compnal


#endif /* COMPNAL_UTILITY_HASH_HPP_ */
