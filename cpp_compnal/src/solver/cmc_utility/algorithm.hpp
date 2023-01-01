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
//  algorithm.hpp
//  compnal
//
//  Created by kohei on 2022/11/23.
//  
//

#ifndef COMPNAL_SOLVER_CMC_UTILITY_ALGORITHM_
#define COMPNAL_SOLVER_CMC_UTILITY_ALGORITHM_

namespace compnal {
namespace solver {
namespace cmc_utility {

enum class Algorithm {
   
   METROPOLIS, // SSF Update
   HEAT_BATH, // SSF Update
   IRKMR, // SSF Update by Suwa Todo method (Reversible Kernel with Minimized Rejection)
   RKMR, // SSF Update by Suwa Todo method (Irreversible Kernel with Minimized Rejection)
   SWENDSEN_WANG, // Cluster Update
   WOLFF // Cluster Update
   
};

template<typename RealType>
auto metropolis_transition = [](const RealType delta_energy, const RealType beta, const RealType dist_real) {
   if (delta_energy <= 0.0 || std::exp(-beta*delta_energy) > dist_real) {
      return true;
   }
   else {
      return false;
   }
};

template<typename RealType>
auto heat_bath_transition = [](const RealType delta_energy, const RealType beta, const RealType dist_real) {
   if (1.0/(1.0 + std::exp(beta*delta_energy)) > dist_real) {
      return true;
   }
   else {
      return false;
   }
};


} // namespace cmc_utility
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CMC_UTILITY_CMC_UPDATER_HPP_ */
