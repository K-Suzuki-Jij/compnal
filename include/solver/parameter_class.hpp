//
//  Copyright 2024 Kohei Suzuki
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
//  parameter_class.hpp
//  compnal
//
//  Created by kohei on 2023/05/04.
//  
//

#pragma once

namespace compnal {
namespace solver {

enum class StateUpdateMethod {
   
   //! @brief Metropolis updater, single spin flip.
   METROPOLIS,
   
   //! @brief Heat bath updater, single spin flip.
   HEAT_BATH,
   
   //! @brief Suwa-Todo updater, single spin flip.
   SUWA_TODO

};

enum class RandomNumberEngine {
  
   //! @brief 32-bit Mersenne Twister
   MT,
   
   //! @brief 64-bit Mersenne Twister
   MT_64,
   
   //! @brief 32-bit Xorshift
   XORSHIFT
   
};

enum class SpinSelectionMethod {
   
   //! @brief Randomly select a spin.
   RANDOM,
   
   //! @brief Select a spin sequentially.
   SEQUENTIAL
   
};

} // namespace solver
} // namespace compnal
