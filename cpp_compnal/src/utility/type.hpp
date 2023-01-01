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
//  type.hpp
//  compnal
//
//  Created by kohei on 2022/08/13.
//  
//

#ifndef COMPNAL_UTILITY_TYPE_HPP_
#define COMPNAL_UTILITY_TYPE_HPP_

#include "./random_number.hpp"
#include <random>

namespace compnal {
namespace utility {

using AnyTupleType = std::vector<std::variant<std::int32_t, std::string>>;

using AnyIndexType = std::variant<std::int32_t, std::string, AnyTupleType>;

using SpinType = std::int8_t;

using RandType = std::mt19937_64;
//using RandType = std::mt19937;
//using RandType = Xorshift;

const std::int32_t DEFAULT_NUM_SWEEPS = 1000;
const std::int32_t DEFAULT_NUM_SAMPLES = 1;
const std::int32_t DEFAULT_NUM_THREADS = 1;

template<typename RealType>
const RealType DEFAULT_BETA = 1;


} // namespace utility
} // namespace compnal

#endif /* COMPNAL_UTILITY_TYPE_HPP_ */
