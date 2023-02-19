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
//  kondo_lattice.hpp
//  compnal
//
//  Created by kohei on 2023/02/04.
//  
//

#ifndef COMPNAL_MODEL_QUANTUM_KONDO_LATTICE_HPP_
#define COMPNAL_MODEL_QUANTUM_KONDO_LATTICE_HPP_

#include "./base_spin_electron.hpp"

namespace compnal {
namespace model {
namespace quantum {


template<class LatticeType, typename RealType>
class KondoLattice: public BaseSpinElectron<LatticeType, RealType> {
   
public:
   using IndexType = typename LatticeType::IndexType;
   
   KondoLattice(const LatticeType &lattice): BaseSpinElectron<LatticeType, RealType>(lattice) {}
   
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
   
   void SetMagneticFieldElectron(const RealType magnetic_field_electron) {
      magnetic_field_electron_ = magnetic_field_electron;
   }
   
   void SetMagneticFieldSpin(const RealType magnetic_field_spin) {
      magnetic_field_spin_ = magnetic_field_spin;
   }
   
   void SetAnisotropy(const RealType anisotropy) {
      anisotropy_ = anisotropy;
   }

   void SetKondoExchangeZ(const RealType kondo_exchange_z) {
      kondo_exchange_z_ = kondo_exchange_z;
   }

   void SetKondoExchangeXY(const RealType kondo_exchange_xy) {
      kondo_exchange_xy_ = kondo_exchange_xy;
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
   
   RealType GetMagneticFieldElectron() const {
      return magnetic_field_electron_;
   }
   
   RealType GetMagneticFieldSpin() const {
      return magnetic_field_spin_;
   }
   
   RealType GetAnisotropy() const {
      return anisotropy_;
   }

   RealType GetKondoExchangeZ() {
      return kondo_exchange_z_;
   }

   RealType GetKondoExchangeXY() {
      return kondo_exchange_xy_;
   }
   
   blas::CRS<RealType> GenarateOnsiteOperatorHam() const {
      return
      magnetic_field_electron_*this->GenerateOnsiteOperatorSzC() +
      onsite_coulomb_*this->GetOnsiteOperatorNCUp()*this->GetOnsiteOperatorNCDown() +
      chemical_potential_*this->GetOnsiteOperatorNC() +
      magnetic_field_spin_*this->GenerateOnsiteOperatorSzL() +
      anisotropy_*this->GetOnsiteOperatorSzL()*this->GetOnsiteOperatorSzL() +
      kondo_exchange_z_*this->GenerateOnsiteOperatorSzL()*this->GenerateOnsiteOperatorSzC() +
      0.5*kondo_exchange_xy_*(this->GenerateOnsiteOperatorSpC()*this->GenerateOnsiteOperatorSmL() + this->GenerateOnsiteOperatorSpL()*this->GenerateOnsiteOperatorSmC());
   }
   
   
   
private:
   //! @brief Hopping energy \f$ t \f$.
   std::vector<RealType> hopping_energy_ = {1.0};
   
   //! @brief The onsite density interactions \f$ U \f$.
   RealType onsite_coulomb_ = 1.0;
   
   std::vector<RealType> intersite_coulomb_ = {};
      
   //! @brief The chemical potential \f$ mu \f$.
   RealType chemical_potential_ = 0.0;
      
   //! @brief The uniaxial anisotropy to the z-direction \f$ D_z\f$.
   RealType anisotropy_ = 0.0;
   
   //! @brief The Kondo exchange coupling along the z-direction \f$J_z \f$.
   RealType kondo_exchange_z_ = 1.0;
   
   //! @brief The Kondo exchange coupling along the x, y-direction \f$J_{xy} \f$.
   RealType kondo_exchange_xy_ = 1.0;
   
   //! @brief The magnetic fields along the z-direction \f$ h_z \f$.
   RealType magnetic_field_electron_ = 0.0;
   RealType magnetic_field_spin_ = 0.0;
   

   
};

} // namespace quantum
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_QUANTUM_KONDO_LATTICE_HPP_ */
