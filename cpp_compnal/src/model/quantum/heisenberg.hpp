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
//  heisenberg.hpp
//  compnal
//
//  Created by kohei on 2022/08/13.
//  
//

#ifndef COMPNAL_MODEL_HEISENBERG_HPP_
#define COMPNAL_MODEL_HEISENBERG_HPP_

#include "./base_spin.hpp"
#include "../../blas/compressed_row_storage.hpp"

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class Heisenberg: public BaseSpin<LatticeType, RealType> {
   
public:
   
   using IndexType = typename LatticeType::IndexType;
   
   Heisenberg(const LatticeType &lattice): BaseSpin<LatticeType, RealType>(lattice) {}
   
   void SetSpinSpinZ(const std::vector<RealType> &spin_spin_z) {
      spin_spin_z_ = spin_spin_z;
   }
   
   void SetSpinSpinXY(const std::vector<RealType> &spin_spin_xy) {
      spin_spin_xy_ = spin_spin_xy;
   }
   
   void SetMagneticField(const RealType magnetic_field) {
      magnetic_field_ = magnetic_field;
   }
   
   void SetAnisotropy(const RealType anisotropy) {
      anisotropy_ = anisotropy;
   }
   
   std::vector<RealType> GetSpinSpinZ() const {
      return spin_spin_z_;
   }
   
   std::vector<RealType> GetSpinSpinXY() const {
      return spin_spin_xy_;
   }
   
   RealType GetMagneticField() const {
      return magnetic_field_;
   }
   
   RealType GetAnisotropy() const {
      return anisotropy_;
   }
   
   blas::CRS<RealType> GenarateOnsiteOperatorHam() const {
      return
      magnetic_field_*this->GetOnsiteOperatorSz() +
      anisotropy_*this->GetOnsiteOperatorSz()*this->GetOnsiteOperatorSz();
   }
   
private:
   //! @brief Hopping energy \f$ t \f$.
   std::vector<RealType> spin_spin_z_ = {1.0};
   
   std::vector<RealType> spin_spin_xy_ = {1.0};
   
   RealType magnetic_field_ = 0.0;
   
   RealType anisotropy_ = 0.0;
   
};

} // namespace quantum
} // namespace model
} // namespace compnal


#endif /* COMPNAL_MODEL_HEISENBERG_HPP_ */
