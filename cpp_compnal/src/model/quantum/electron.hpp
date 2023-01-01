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
//  electron.hpp
//  compnal
//
//  Created by kohei on 2022/12/31.
//  
//

#ifndef COMPNAL_MODEL_QUANTUM_ELECTRON_HPP_
#define COMPNAL_MODEL_QUANTUM_ELECTRON_HPP_

#include "../../lattice/all.hpp"
#include "../../blas/all.hpp"

#include <vector>

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class Electron {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
   //! @brief Alias of compressed row strage (CRS) with RealType.
   using CRS = blas::CRS<RealType>;
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief The type of conserved quantum number (total electron, total sz) pair.
   using CQNType = std::pair<std::int32_t, std::int32_t>;
   
   //! @brief The linear interaction type.
   using LinearType = RealType;
   
   //! @brief The quadratic interaction type.
   using QuadraticType = RealType;
   
   //! @brief Constructor for Electron class.
   //! @param lattice The lattice type.
   //! @param linear The linear interaction.
   //! @param quadratic The quadratic interaction.
   Electron(const LatticeType &lattice,
            const CQNType conserved_quantum_number,
            const LinearType linear,
            const QuadraticType quadratic):
   lattice_(lattice), conserved_quantum_number_(conserved_quantum_number), linear_(linear), quadratic_(quadratic) {
      SetOnsiteOperator();
   }
   
   //! @brief Generate the annihilation operator for the electrons with the up spin \f$ \hat{c}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{c}_{\uparrow}\f$.
   CRS GenerateOnsiteOperatorCUp() const {
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      const RealType val = RealType{1.0};
      const int dim_onsite = 4;
      CRS matrix(dim_onsite, dim_onsite);
      for (int row = 0; row < dim_onsite; row++) {
         for (int col = 0; col < dim_onsite; col++) {
            if ((col == 1 && row == 0) || (col == 3 && row == 2)) {
               matrix.col.push_back(col);
               matrix.val.push_back(val);
            }
         }
         matrix.row[row + 1] = matrix.col.size();
      }
      
      matrix.tag = blas::CRSTag::FERMION;
      
      return matrix;
   }
   
   //! @brief Generate the annihilation operator for the electrons with the down spin \f$ \hat{c}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{c}_{\downarrow}\f$.
   CRS GenerateOnsiteOperatorCDown() const {
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      const RealType val = RealType{1.0};
      const int dim_onsite = 4;
      CRS matrix(dim_onsite, dim_onsite);
      for (int row = 0; row < dim_onsite; row++) {
         for (int col = 0; col < dim_onsite; col++) {
            if (col == 2 && row == 0) {
               matrix.col.push_back(col);
               matrix.val.push_back(val);
            } else if (col == 3 && row == 1) {
               matrix.col.push_back(col);
               matrix.val.push_back(-val);
            }
         }
         matrix.row[row + 1] = matrix.col.size();
      }
      
      matrix.tag = blas::CRSTag::FERMION;
      
      return matrix;
   }
   
   //! @brief Generate the creation operator for the electrons with the up spin
   //! \f$ \hat{c}^{\dagger}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{c}^{\dagger}_{\uparrow}\f$.
   CRS GenerateOnsiteOperatorCUpDagger() { return blas::GenerateTransposedMatrix(GenerateOnsiteOperatorCUp()); }
   
   //! @brief Generate the creation operator for the electrons with the down spin
   //! \f$ \hat{c}^{\dagger}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{c}^{\dagger}_{\downarrow}\f$.
   CRS GenerateOnsiteOperatorCDownDagger() { return blas::GenerateTransposedMatrix(GenerateOnsiteOperatorCDown()); }
   
   //! @brief Generate the number operator for the electrons with the up spin
   //! \f$ \hat{n}_{\uparrow}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{n}_{\uparrow}\f$.
   CRS GenerateOnsiteOperatorNCUp() { return GenerateOnsiteOperatorCUpDagger() * GenerateOnsiteOperatorCUp(); }
   
   //! @brief Generate the number operator for the electrons with the down spin
   //! \f$ \hat{n}_{\downarrow}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{n}_{\downarrow}\f$.
   CRS GenerateOnsiteOperatorNCDown() { return GenerateOnsiteOperatorCDownDagger() * GenerateOnsiteOperatorCDown(); }
   
   //! @brief Generate the number operator for the electrons
   //! \f$ \hat{n}=\hat{n}_{\uparrow} + \hat{n}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{n}\f$.
   CRS GenerateOnsiteOperatorNC() { return GenerateOnsiteOperatorNCUp() + GenerateOnsiteOperatorNCDown(); }
   
   //! @brief Generate the spin operator for the x-direction for the electrons
   //! \f$ \hat{s}^{x}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} +
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$.
   //! @return The matrix of \f$ \hat{s}^{x}\f$.
   CRS GenerateOnsiteOperatorSx() { return RealType{0.5} * (GenerateOnsiteOperatorSp() + GenerateOnsiteOperatorSm()); }
   
   //! @brief Generate the spin operator for the y-direction for the electrons
   //! \f$ i\hat{s}^{y}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$. Here \f$ i=\sqrt{-1}\f$ is the the imaginary unit.
   //! @return The matrix of \f$ i\hat{s}^{y}\f$.
   CRS GenerateOnsiteOperatoriSy() {
      return RealType{0.5} * (GenerateOnsiteOperatorSp() - GenerateOnsiteOperatorSm());
   }
   
   //! @brief Generate the spin operator for the z-direction for the electrons
   //! \f$ \hat{s}^{z}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow})\f$.
   //! @return The matrix of \f$ \hat{s}^{z}\f$.
   CRS GenerateOnsiteOperatorSz() {
      return RealType{0.5} * (GenerateOnsiteOperatorNCUp() - GenerateOnsiteOperatorNCDown());
   }
   
   //! @brief Generate the raising operator for spin of the electrons
   //! \f$ \hat{s}^{+}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{s}^{+}\f$.
   CRS GenerateOnsiteOperatorSp() { return GenerateOnsiteOperatorCUpDagger() * GenerateOnsiteOperatorCDown(); }
   
   //! @brief Generate the lowering operator for spin of the electrons
   //! \f$ \hat{s}^{-}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{s}^{-}\f$.
   CRS GenerateOnsiteOperatorSm() { return GenerateOnsiteOperatorCDownDagger() * GenerateOnsiteOperatorCUp(); }
   
   
private:
   //! @brief The linear interaction.
   const LatticeType lattice_;
   
   //! @brief The linear interaction.
   const LinearType linear_ = 0;
   
   //! @brief The quadratic interaction.
   const QuadraticType quadratic_ = 0;
   
   //! @brief Pair of the total electron \f$ \langle\hat{N}_{\rm e}\rangle\f$ and the total sz \f$ \langle\hat{S}^{z}_{\rm tot}\rangle\f$.
   CQNType conserved_quantum_number_ = {0, 0};
   
   //! @brief The dimension of the local Hilbert space, 4.
   const int dim_onsite_ = 4;
   
   //! @brief The annihilation operator for the electrons with the up spin \f$ \hat{c}_{\uparrow}\f$.
   CRS onsite_operator_c_up_;
   
   //! @brief The annihilation operator for the electrons with the down spin \f$ \hat{c}_{\downarrow}\f$.
   CRS onsite_operator_c_down_;
   
   //! @brief The creation operator for the electrons with the up spin.
   //! \f$ \hat{c}^{\dagger}_{\uparrow}\f$.
   CRS onsite_operator_c_up_dagger_;
   
   //! @brief The creation operator for the electrons with the down spin.
   //! \f$ \hat{c}^{\dagger}_{\downarrow}\f$.
   CRS onsite_operator_c_down_dagger_;
   
   //! @brief The number operator for the electrons with the up spin
   //! \f$ \hat{n}_{\uparrow}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow}\f$.
   CRS onsite_operator_nc_up_;
   
   //! @brief The number operator for the electrons with the down spin
   //! \f$ \hat{n}_{\downarrow}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow}\f$.
   CRS onsite_operator_nc_down_;
   
   //! @brief The number operator for the electrons
   //! \f$ \hat{n}=\hat{n}_{\uparrow} + \hat{n}_{\downarrow}\f$.
   CRS onsite_operator_nc_;
   
   //! @brief The spin operator for the x-direction for the electrons
   //! \f$ \hat{s}^{x}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} +
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$.
   CRS onsite_operator_sx_;
   
   //! @brief The spin operator for the y-direction for the electrons
   //! \f$ i\hat{s}^{y}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$. Here \f$ i=\sqrt{-1}\f$ is the the imaginary unit.
   CRS onsite_operator_isy_;
   
   //! @brief The spin operator for the z-direction for the electrons
   //! \f$ \hat{s}^{z}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow})\f$.
   CRS onsite_operator_sz_;
   
   //! @brief The raising operator for spin of the electrons
   //! \f$ \hat{s}^{+}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow}\f$.
   CRS onsite_operator_sp_;
   
   //! @brief The lowering operator for spin of the electrons
   //! \f$ \hat{s}^{-}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow}\f$.
   CRS onsite_operator_sm_;
   
   //! @brief Set onsite operators.
   void SetOnsiteOperator() {
      onsite_operator_c_up_ = GenerateOnsiteOperatorCUp();
      onsite_operator_c_down_ = GenerateOnsiteOperatorCDown();
      onsite_operator_c_up_dagger_ = GenerateOnsiteOperatorCUpDagger();
      onsite_operator_c_down_dagger_ = GenerateOnsiteOperatorCDownDagger();
      onsite_operator_nc_up_ = GenerateOnsiteOperatorNCUp();
      onsite_operator_nc_down_ = GenerateOnsiteOperatorNCDown();
      onsite_operator_nc_ = GenerateOnsiteOperatorNC();
      onsite_operator_sx_ = GenerateOnsiteOperatorSx();
      onsite_operator_isy_ = GenerateOnsiteOperatoriSy();
      onsite_operator_sz_ = GenerateOnsiteOperatorSz();
      onsite_operator_sp_ = GenerateOnsiteOperatorSp();
      onsite_operator_sm_ = GenerateOnsiteOperatorSm();
   }
   
   
   
};

} // quantum
} // namespace model
} // namespace compnal


#endif /* COMPNAL_MODEL_QUANTUM_ELECTRON_HPP_ */
