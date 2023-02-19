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

#include "./base_electron.hpp"
#include "../../blas/compressed_row_storage.hpp"

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class Hubbard: public BaseElectron<LatticeType, RealType> {
   
public:
   using IndexType = typename LatticeType::IndexType;
   
   Hubbard(const LatticeType &lattice): BaseElectron<LatticeType, RealType>(lattice) {}
      
   void SetHoppingEnergy(const std::vector<RealType> &hopping_energy) {
      hopping_energy_ = hopping_energy;
   }
   
   void SetOnsiteCoulomb(const RealType onsite_coulomb) {
      onsite_coulomb_ = onsite_coulomb;
   }
   
   void SetIntersiteCoulomb(const std::vector<RealType> &intersite_coulomb) {
      intersite_coulomb_ = intersite_coulomb;
   }
   
   void SetChemicalPotential(const RealType chemical_potential) {
      chemical_potential_ = chemical_potential;
   }
   
   void SetMagneticField(const RealType magnetic_field) {
      magnetic_field_ = magnetic_field;
   }
   
   std::vector<RealType> GetHoppingEnergy() const {
      return hopping_energy_;
   }
   
   RealType GetOnsiteCoulomb() const {
      return onsite_coulomb_;
   }
   
   std::vector<RealType> GetIntersiteCoulomb() const {
      return intersite_coulomb_;
   }
   
   RealType GetChemicalPotential() const {
      return chemical_potential_;
   }
   
   RealType GetMagneticField() const {
      return magnetic_field_;
   }
   
   blas::CRS<RealType> GenarateOnsiteOperatorHam() const {
      return
      magnetic_field_*this->GetOnsiteOperatorSz() +
      onsite_coulomb_*this->GetOnsiteOperatorNCUp()*this->GetOnsiteOperatorNCDown() +
      chemical_potential_*this->GetOnsiteOperatorNC();
   }
   
private:
   //! @brief Hopping energy \f$ t \f$.
   std::vector<RealType> hopping_energy_ = {1.0};
   
   //! @brief The onsite density interactions \f$ U \f$.
   RealType onsite_coulomb_ = 1.0;
   
   std::vector<RealType> intersite_coulomb_ = {};
      
   //! @brief The chemical potential \f$ mu \f$.
   RealType chemical_potential_ = 0.0;
   
   RealType magnetic_field_ = 0.0;
   
};

} // namespace quantum
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_HUBBARD_HPP_ */
