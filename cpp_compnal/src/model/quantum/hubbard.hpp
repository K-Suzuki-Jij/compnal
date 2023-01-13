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
//  hubbard.hpp
//  compnal
//
//  Created by kohei on 2022/08/13.
//  
//

#ifndef COMPNAL_MODEL_HUBBARD_HPP_
#define COMPNAL_MODEL_HUBBARD_HPP_

#include "./electron.hpp"

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class Hubbard: public Electron<LatticeType, RealType> {
   
public:
   Hubbard(const LatticeType &lattice): Electron<LatticeType, RealType>(lattice) {}
      
   void SetHoppingEnergy(const RealType hoppin_genergy) {
      hoppin_genergy_ = hoppin_genergy;
   }
   
   void SetOnsiteCoulomb(const RealType onsite_coulomb) {
      onsite_coulomb_ = onsite_coulomb;
   }
   
   void SetChemicalPotential(const RealType chemical_potential) {
      chemical_potential_ = chemical_potential;
   }
   
private:
   //! @brief Hopping energy \f$ t \f$.
   RealType hoppin_genergy_ = 1.0;
   
   //! @brief The onsite density interactions \f$ U \f$.
   RealType onsite_coulomb_ = 1.0;
      
   //! @brief The chemical potential \f$ mu \f$.
   RealType chemical_potential_ = 0.0;
   
   
   
};

} // namespace quantum
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_HUBBARD_HPP_ */
