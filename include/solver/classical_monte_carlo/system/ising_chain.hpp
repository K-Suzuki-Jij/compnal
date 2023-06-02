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
//  ising_chain.hpp
//  compnal
//
//  Created by kohei on 2023/05/06.
//  
//

#ifndef COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CHAIN_HPP_
#define COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CHAIN_HPP_

#include "template_system.hpp"
#include "../../../model/classical/ising.hpp"
#include "../../../lattice/chain.hpp"
#include "../../../lattice/boundary_condition.hpp"

namespace compnal {
namespace solver {
namespace classical_monte_carlo {

template<class RandType>
class System<model::classical::Ising<lattice::Chain>, RandType> {
   using ModelType = model::classical::Ising<lattice::Chain>;
   
private:
   const std::int32_t system_size_ = 0;
   const lattice::BoundaryCondition bc_ = lattice::BoundaryCondition::NONE;
   const double linear_ = 0;
   const double quadratic_ = 0;
   std::vector<typename ModelType::PHQType> sample_;
   
   
};

} // namespace classical_monte_carlo
} // namespace solver
} // namespace compnal


#endif /* COMPNAL_SOLVER_CLASSICAL_MONTE_CARLO_ISING_CHAIN_HPP_ */
