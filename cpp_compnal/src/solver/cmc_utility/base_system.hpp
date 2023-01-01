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
//  base_system.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_BASE_SYSTEM_HPP_
#define COMPNAL_SOLVER_CMC_UTILITY_BASE_SYSTEM_HPP_

#include "../../utility/all.hpp"

namespace compnal {
namespace solver {
namespace cmc_utility {

template<class ModelType>
class CMCSystem;

class CMCBaseIsingSystem {
   
protected:
   std::vector<utility::SpinType> GenerateRandomSpin(const std::uint64_t seed, const std::int32_t system_size) const {
      std::vector<utility::SpinType> sample(system_size);
      std::uniform_int_distribution<utility::SpinType> dist(0, 1);
      utility::RandType random_number_engine(seed);
      for (std::size_t i = 0; i < sample.size(); i++) {
         sample[i] = 2*dist(random_number_engine) - 1;
      }
      return sample;
   }
   
};

} // namespace cmc_utility
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CMC_UTILITY_BASE_SYSTEM_HPP_ */
