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
//  random.hpp
//  compnal
//
//  Created by kohei on 2023/05/04.
//  
//

#ifndef COMPNAL_UTILITY_RANDOM_HPP_
#define COMPNAL_UTILITY_RANDOM_HPP_

#include <random>

namespace compnal {
namespace utility {

//! @brief Class for the random number engine using Xorshift.
class Xorshift {
public:
   //! @brief Result type.
   using result_type = std::uint_fast32_t;
   
   //! @brief The minimum value generated by Xorshift.
   static constexpr std::uint32_t min() { return std::uint32_t{0}; }
   
   //! @brief The maximum value generated by Xorshift.
   static constexpr std::uint32_t max() { return UINT_MAX; }
   
   //! @brief Random number engine using Xorshift
   std::uint32_t operator()() {
      std::uint32_t t = x_ ^ (x_ << 11);
      x_ = y_;
      y_ = z_;
      z_ = w_;
      return w_ = (w_ ^ (w_ >> 19)) ^ (t ^ (t >> 8));
   }
   
   //! @brief Constructor of Xorshift.
   Xorshift() {
      std::random_device rd;
      w_ = rd();
   }
   
   //! @brief Constructor of Xorshift.
   //! @param seed The seed for Xorshift.
   Xorshift(const std::uint32_t seed) { w_ = seed; }
   
private:
   //! @brief One of the seed, which is initialized as constant value.
   std::uint32_t x_ = std::uint32_t{123456789};
   
   //! @brief One of the seed, which is initialized as constant value.
   std::uint32_t y_ = std::uint32_t{362436069};
   
   //! @brief One of the seed, which is initialized as constant value.
   std::uint32_t z_ = std::uint32_t{521288629};
   
   //! @brief One of the seed.
   std::uint32_t w_;
};


} // namespace utility
} // namespace compnal


#endif /* COMPNAL_UTILITY_RANDOM_HPP_ */
