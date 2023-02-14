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
//  base_spin_multi_electron.hpp
//  compnal
//
//  Created by kohei on 2023/02/09.
//  
//

#ifndef COMPNAL_MODEL_QUANTUM_BASE_SPIN_MULTI_ELECTRON_HPP_
#define COMPNAL_MODEL_QUANTUM_BASE_SPIN_MULTI_ELECTRON_HPP_

#include "../../lattice/all.hpp"
#include "../../blas/all.hpp"
#include "../../utility/hash.hpp"

#include <omp.h>
#include <vector>

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class BaseSpinMultiElectron {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
   //! @brief Alias of compressed row strage (CRS) with RealType.
   using CRS = blas::CRS<RealType>;
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief The type of conserved quantum number (total electron, twice the number of total sz) pair.
   using CQNType = std::pair<std::vector<std::int32_t>, std::int32_t>;
   
   using CQNHash = utility::PairVecHash;
   
   BaseSpinMultiElectron(const LatticeType &lattice): lattice_(lattice) {
      SetOnsiteOperator();
   }
   
   void SetTotalSz(const double total_sz) {
      if (std::floor(2*total_sz) != 2*total_sz) {
         std::stringstream ss;
         ss << "The input number " << total_sz << " is not half-integer." << std::endl;
         throw std::runtime_error(ss.str());
      }
      conserved_quantum_number_.second = 2*total_sz;
   }
   
   void SetTotalElectron(const std::vector<std::int32_t> &total_electron_list) {
      for (const auto &total_electron: total_electron_list) {
         if (total_electron < 0) {
            std::stringstream ss;
            ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
            ss << "total_electron must be a non-negative integer" << std::endl;
            throw std::runtime_error(ss.str());
         }
      }
      if (conserved_quantum_number_.first.size() != total_electron_list.size()) {
         conserved_quantum_number_.first = total_electron_list;
         num_orbital_ = static_cast<std::int32_t>(conserved_quantum_number_.first.size());
         dim_onsite_all_electrons_ = static_cast<std::int32_t>(utility::CalculatePower(dim_onsite_electron_, num_orbital_));
         dim_onsite_ = dim_onsite_lspin_*dim_onsite_all_electrons_;
         SetOnsiteOperator();
      }
   }
   
   void SetMagnitudeLSpin(const double magnitude_lspin) {
      if (magnitude_lspin <= 0) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Please set magnitude_spin > 0" << std::endl;
         throw std::runtime_error(ss.str());
      }
      if (std::floor(2*magnitude_lspin) != 2*magnitude_lspin) {
         std::stringstream ss;
         ss << "The input number " << magnitude_lspin << " is not half-integer." << std::endl;
         throw std::runtime_error(ss.str());
      }
      if (magnitude_2lspin_ != 2*magnitude_lspin) {
         magnitude_2lspin_ = 2*magnitude_lspin;
         dim_onsite_lspin_ = magnitude_2lspin_ + 1;
         dim_onsite_ = dim_onsite_lspin_*dim_onsite_all_electrons_;
         SetOnsiteOperator();
      }
   }
   
   void PrintBasisOnsite() const {
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         std::vector<std::string> b_ele;
         for (std::int32_t o = 0; o < static_cast<std::int32_t>(conserved_quantum_number_.first.size()); ++o) {
            if (CalculateBasisOnsiteElectron(row, o) == 0) {
               b_ele.push_back("|vac>");
            }
            else if (CalculateBasisOnsiteElectron(row, o) == 1) {
               b_ele.push_back("| ↑ >");
            }
            else if (CalculateBasisOnsiteElectron(row, o) == 2) {
               b_ele.push_back("| ↓ >");
            }
            else if (CalculateBasisOnsiteElectron(row, o) == 3) {
               b_ele.push_back("|↑↓ >");
            }
            else {
               std::stringstream ss;
               ss << "Unknown error detected in " << __FUNCTION__ << " at " << __LINE__ << std::endl;
               throw std::runtime_error(ss.str());
            }
         }
         std::cout << "row " << std::setw(2) << row << ": ";
         for (const auto &it : b_ele) {
            std::cout << it;
         }
         std::cout << "|Sz=" << std::showpos << 0.5*magnitude_2lspin_ - CalculateBasisOnsiteLSpin(row) << ">" << std::noshowpos << std::endl;
      }
   }
   
   static bool ValidateQNumber(const std::int32_t system_size,
                               const std::int32_t magnitude_2lspin,
                               const std::vector<std::int32_t> &total_electron_list,
                               const std::int32_t total_2sz) {
      const std::int32_t total_electron = std::accumulate(total_electron_list.begin(), total_electron_list.end(), 0);
      
      for (const auto &total_electron_per_orbital: total_electron_list) {
         if (total_electron_per_orbital < 0 || 2*system_size < total_electron_per_orbital) {
            return false;
         }
      }
      
      const bool c1 = ((total_electron + system_size*magnitude_2lspin - total_2sz)%2 == 0);
      const bool c2 = (-(total_electron + system_size*magnitude_2lspin) <= total_2sz);
      const bool c3 = (total_2sz <= total_electron + system_size*magnitude_2lspin);
      
      if (!c1 && !c2 && !c3) {
         return false;
      }
      
      return true;
   }
   
   
   static std::int64_t CalculateTargetDim(const std::int32_t system_size,
                                          const std::int32_t magnitude_2lspin,
                                          const std::vector<std::int32_t> &total_electron_list,
                                          const std::int32_t total_2sz) {
      if (!ValidateQNumber(system_size, magnitude_2lspin, total_electron_list, total_2sz)) {
         return 0;
      }
      
      std::vector<std::vector<std::vector<std::int32_t>>> electron_configuration_list;
      std::vector<std::int32_t> length_list;
      std::int64_t length = 1;
      for (const auto num_electron : total_electron_list) {
         const auto electron_configuration = GenerateElectronConfigurations(system_size, num_electron);
         electron_configuration_list.push_back(electron_configuration);
         length_list.push_back(static_cast<std::int32_t>(electron_configuration[0].size()));
         length *= electron_configuration[0].size();
      }
      const std::vector<std::vector<std::int64_t>> binom = utility::GenerateBinomialTable(system_size);
      std::int64_t dim = 0;
      for (std::int64_t i = 0; i < length; ++i) {
         std::int32_t electron_2sz = 0;
         std::int64_t electron_dim = 1;
         for (std::size_t j = 0; j < length_list.size(); ++j) {
            std::int64_t prod = 1;
            for (std::size_t k = j + 1; k < length_list.size(); ++k) {
               prod *= length_list[k];
            }
            const std::size_t index = (i / prod) % length_list[j];
            const std::int32_t n_up_down = electron_configuration_list[j][0][index];
            const std::int32_t n_up = electron_configuration_list[j][1][index];
            const std::int32_t n_down = electron_configuration_list[j][2][index];
            electron_2sz += n_up - n_down;
            electron_dim *= binom[system_size][n_up]*binom[system_size - n_up][n_down]*binom[system_size - n_up - n_down][n_up_down];
         }
         const std::int32_t spin_2sz = total_2sz - electron_2sz;
         if (BaseSpin<LatticeType, RealType>::ValidateQNumber(system_size, magnitude_2lspin, spin_2sz)) {
            dim += electron_dim*BaseSpin<LatticeType, RealType>::CalculateTargetDim(system_size, magnitude_2lspin, spin_2sz);
         }
      }
      return dim;
   }
   
   template<typename IntegerType>
   CQNType CalculateQNumber(const IntegerType row, const IntegerType col) const {
      static_assert(std::is_integral<IntegerType>::value, "Template parameter IntegerType must be integer type");
      const auto &total_electron_list = conserved_quantum_number_.first;
      std::vector<std::int32_t> target_total_electron_list(num_orbital_);
      std::int32_t target_total_2sz = 0;
      std::int32_t temp_row_electron = row/dim_onsite_lspin_;
      std::int32_t temp_col_electron = col/dim_onsite_lspin_;
      for (std::int32_t orbital = 0; orbital < num_orbital_; ++orbital) {
         const std::int32_t row_electron = temp_row_electron%dim_onsite_electron_;
         const std::int32_t col_electron = temp_col_electron%dim_onsite_electron_;
         if (row_electron == col_electron && 0 <= row_electron && row_electron < 4 && 0 <= col_electron && col_electron < 4) {
            target_total_electron_list[num_orbital_ - orbital - 1] = total_electron_list[orbital];
         }
         else if (row_electron == 0 && col_electron == 1) {
            target_total_electron_list[num_orbital_ - orbital - 1] = -1 + total_electron_list[orbital];
            target_total_2sz += -1;
         }
         else if (row_electron == 0 && col_electron == 2) {
            target_total_electron_list[num_orbital_ - orbital - 1] = -1 + total_electron_list[orbital];
            target_total_2sz += +1;
         }
         else if (row_electron == 0 && col_electron == 3) {
            target_total_electron_list[num_orbital_ - orbital - 1] = -2 + total_electron_list[orbital];
         }
         else if (row_electron == 1 && col_electron == 0) {
            target_total_electron_list[num_orbital_ - orbital - 1] = +1 + total_electron_list[orbital];
            target_total_2sz += +1;
         }
         else if (row_electron == 1 && col_electron == 2) {
            target_total_electron_list[num_orbital_ - orbital - 1] = total_electron_list[orbital];
            target_total_2sz += +2;
         }
         else if (row_electron == 1 && col_electron == 3) {
            target_total_electron_list[num_orbital_ - orbital - 1] = -1 + total_electron_list[orbital];
            target_total_2sz += +1;
         }
         else if (row_electron == 2 && col_electron == 0) {
            target_total_electron_list[num_orbital_ - orbital - 1] = +1 + total_electron_list[orbital];
            target_total_2sz += -1;
         }
         else if (row_electron == 2 && col_electron == 1) {
            target_total_electron_list[num_orbital_ - orbital - 1] = total_electron_list[orbital];
            target_total_2sz += -2;
         }
         else if (row_electron == 2 && col_electron == 3) {
            target_total_electron_list[num_orbital_ - orbital - 1] = -1 + total_electron_list[orbital];
            target_total_2sz += -1;
         }
         else if (row_electron == 3 && col_electron == 0) {
            target_total_electron_list[num_orbital_ - orbital - 1] = 2 + total_electron_list[orbital];
         }
         else if (row_electron == 3 && col_electron == 1) {
            target_total_electron_list[num_orbital_ - orbital - 1] = 1 + total_electron_list[orbital];
            target_total_2sz += -1;
         }
         else if (row_electron == 3 && col_electron == 2) {
            target_total_electron_list[num_orbital_ - orbital - 1] = 1 + total_electron_list[orbital];
            target_total_2sz += +1;
         }
         else {
            std::stringstream ss;
            ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
            ss << "The dimenstion of the matrix must be 4";
            throw std::runtime_error(ss.str());
         }
         temp_row_electron /= dim_onsite_electron_;
         temp_col_electron /= dim_onsite_electron_;
      }
      target_total_2sz += 2*col%dim_onsite_lspin_ - 2*row%dim_onsite_lspin_ + conserved_quantum_number_.second;
      return {target_total_electron_list, target_total_2sz};
   }
   
   CRS CreateOnsiteOperatorCUp(const std::int32_t orbital) const {
      
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         std::int32_t num_electron = 0;
         for (std::int32_t o = 0; o < orbital; ++o) {
            const std::int32_t basis_electron_onsite = CalculateBasisOnsiteElectron(row, o);
            num_electron += CalculateNumElectronFromElectronBasis(basis_electron_onsite);
         }
         std::int32_t sign = 1;
         if (num_electron%2 == 1) {
            sign = -1;
         }
         
         for (std::int32_t col = 0; col < dim_onsite_; col++) {
            bool c_1 = true;
            for (std::int32_t o = 0; o < num_orbital_; ++o) {
               const std::int32_t basis_1 = CalculateBasisOnsiteElectron(row, o);
               const std::int32_t basis_2 = CalculateBasisOnsiteElectron(col, o);
               if (o != orbital && basis_1 != basis_2) {
                  c_1 = false;
                  break;
               }
            }
            
            std::int32_t basis_row_electron = CalculateBasisOnsiteElectron(row, orbital);
            std::int32_t basis_col_electron = CalculateBasisOnsiteElectron(col, orbital);
            
            const bool c_2 = (basis_row_electron == 0 && basis_col_electron == 1);
            const bool c_3 = (basis_row_electron == 2 && basis_col_electron == 3);
            
            if (c_1 && row%dim_onsite_lspin_ == col%dim_onsite_lspin_ && (c_2 || c_3)) {
               matrix.val.push_back(sign);
               matrix.col.push_back(col);
            }
         }
         matrix.row[row + 1] = matrix.col.size();
      }
      matrix.tag = blas::CRSTag::FERMION;
      return matrix;
   }
   
   CRS CreateOnsiteOperatorCDown(const std::int32_t orbital) const {
      
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         std::int32_t num_electron = 0;
         for (std::int32_t o = 0; o < orbital; ++o) {
            const std::int32_t basis_electron_onsite = CalculateBasisOnsiteElectron(row, o);
            num_electron += CalculateNumElectronFromElectronBasis(basis_electron_onsite);
         }
         std::int32_t sign_1 = 1;
         if (num_electron%2 == 1) {
            sign_1 = -1;
         }
         
         for (std::int32_t col = 0; col < dim_onsite_; col++) {
            bool c_1 = true;
            for (std::int32_t o = 0; o < num_orbital_; ++o) {
               const std::int32_t basis_1 = CalculateBasisOnsiteElectron(row, o);
               const std::int32_t basis_2 = CalculateBasisOnsiteElectron(col, o);
               if (o != orbital && basis_1 != basis_2) {
                  c_1 = false;
                  break;
               }
            }
            
            std::int32_t basis_row_electron = CalculateBasisOnsiteElectron(row, orbital);
            std::int32_t basis_col_electron = CalculateBasisOnsiteElectron(col, orbital);
            
            const bool c_2 = (basis_row_electron == 0 && basis_col_electron == 2);
            const bool c_3 = (basis_row_electron == 1 && basis_col_electron == 3);
            
            std::int32_t sign_2 = 1;
            if (c_3) {
               sign_2 = -1;
            }
            
            if (c_1 && row%dim_onsite_lspin_ == col%dim_onsite_lspin_ && (c_2 || c_3)) {
               matrix.val.push_back(sign_1*sign_2);
               matrix.col.push_back(col);
            }
         }
         matrix.row[row + 1] = matrix.col.size();
      }
      matrix.tag = blas::CRSTag::FERMION;
      return matrix;
   }
   
   CRS CreateOnsiteOperatorSzL() const {
      
      CRS matrix(dim_onsite_, dim_onsite_);
      const RealType val = 0.5*magnitude_2lspin_;
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         for (std::int32_t col = 0; col < dim_onsite_; ++col) {
            const std::int32_t basis_row_lspin = row%dim_onsite_lspin_;
            const std::int32_t basis_col_lspin = col%dim_onsite_lspin_;
            const std::int32_t basis_row_total_electron = row / dim_onsite_lspin_;
            const std::int32_t basis_col_total_electron = col / dim_onsite_lspin_;
            if (basis_row_total_electron == basis_col_total_electron && basis_row_lspin == basis_col_lspin) {
               matrix.val.push_back(val - basis_row_lspin);
               matrix.col.push_back(col);
            }
         }
         matrix.row.push_back(matrix.col.size());
      }
      return matrix;
   }
   
   CRS CreateOnsiteOperatorSpL() const {
      
      CRS matrix(dim_onsite_, dim_onsite_);
      const RealType val = 0.5*magnitude_2lspin_ +1;
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         for (std::int32_t col = 0; col < dim_onsite_; ++col) {
            const std::int32_t a = row%dim_onsite_lspin_ + 1;
            const std::int32_t b = col%dim_onsite_lspin_ + 1;
            const std::int32_t basis_row_total_electron = row/dim_onsite_lspin_;
            const std::int32_t basis_col_total_electron = col/dim_onsite_lspin_;
            if (basis_row_total_electron == basis_col_total_electron && a + 1 == b) {
               matrix.val.push_back(std::sqrt(val*(a + b - 1) - a*b));
               matrix.col.push_back(col);
            }
         }
         matrix.row.push_back(matrix.col.size());
      }
      return matrix;
   }
   
   
   CRS CreateOnsiteOperatorCUpDagger(const std::int32_t orbital) const {
      return blas::GenerateTransposedMatrix(CreateOnsiteOperatorCUp(orbital));
   }
   
   CRS CreateOnsiteOperatorCDownDagger(const std::int32_t orbital) const {
      return blas::GenerateTransposedMatrix(CreateOnsiteOperatorCDown(orbital));
   }
   
   CRS CreateOnsiteOperatorNCUp(const std::int32_t orbital) const {
      return CreateOnsiteOperatorCUpDagger(orbital)*CreateOnsiteOperatorCUp(orbital);
   }
   
   CRS CreateOnsiteOperatorNCDown(const std::int32_t orbital) const {
      return CreateOnsiteOperatorCDownDagger(orbital)*CreateOnsiteOperatorCDown(orbital);
   }
   
   CRS CreateOnsiteOperatorNC(const std::int32_t orbital) const {
      return CreateOnsiteOperatorNCUp(orbital) + CreateOnsiteOperatorNCDown(orbital);
   }
   
   CRS CreateOnsiteOperatorNCTot() const {
      CRS out(dim_onsite_, dim_onsite_);
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         out = out + CreateOnsiteOperatorNC(o);
      }
      return out;
   }
   
   CRS CreateOnsiteOperatorSxC(const std::int32_t orbital) const {
      return RealType{0.5}*(CreateOnsiteOperatorSpC(orbital) + CreateOnsiteOperatorSmC(orbital));
   }
   
   CRS CreateOnsiteOperatoriSyC(const std::int32_t orbital) const {
      return RealType{0.5}*(CreateOnsiteOperatorSpC(orbital) - CreateOnsiteOperatorSmC(orbital));
   }
   
   CRS CreateOnsiteOperatorSzC(const std::int32_t orbital) const {
      return RealType{0.5}*(CreateOnsiteOperatorNCUp(orbital) - CreateOnsiteOperatorNCDown(orbital));
   }
   
   CRS CreateOnsiteOperatorSpC(const std::int32_t orbital) const {
      return CreateOnsiteOperatorCUpDagger(orbital)*CreateOnsiteOperatorCDown(orbital);
   }
   
   CRS CreateOnsiteOperatorSmC(const std::int32_t orbital) const {
      return CreateOnsiteOperatorCDownDagger(orbital)*CreateOnsiteOperatorCUp(orbital);
   }
   
   CRS CreateOnsiteOperatorSmL() const {
      return blas::GenerateTransposedMatrix(CreateOnsiteOperatorSpL());
   }
   
   CRS CreateOnsiteOperatorSxL() const {
      return RealType{0.5}*(CreateOnsiteOperatorSpL() + CreateOnsiteOperatorSmL());
   }
   
   CRS CreateOnsiteOperatoriSyL() const {
      return RealType{0.5}*(CreateOnsiteOperatorSpL() - CreateOnsiteOperatorSmL());
   }
   
   CRS CreateOnsiteOperatorSCSL(const std::int32_t orbital) const {
      const CRS spc = CreateOnsiteOperatorSpC(orbital);
      const CRS smc = CreateOnsiteOperatorSmC(orbital);
      const CRS szc = CreateOnsiteOperatorSzC(orbital);
      const CRS spl = CreateOnsiteOperatorSpL();
      const CRS sml = CreateOnsiteOperatorSmL();
      const CRS szl = CreateOnsiteOperatorSzL();
      return szc*szl + RealType{0.5}*(spc*sml + smc*spl);
   }
   
   CRS CreateOnsiteOperatorSCSLTot() const {
      const CRS spl = CreateOnsiteOperatorSpL();
      const CRS sml = CreateOnsiteOperatorSmL();
      const CRS szl = CreateOnsiteOperatorSzL();
      CRS out(szl.row_dim, szl.col_dim);
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         const CRS spc = CreateOnsiteOperatorSpC(o);
         const CRS smc = CreateOnsiteOperatorSmC(o);
         const CRS szc = CreateOnsiteOperatorSzC(o);
         out += szc*szl + RealType{0.5}*(spc*sml + smc*spl);
      }
      return out;
   }
   
   CRS CreateOnsiteOperatorSp() const {
      CRS out = CreateOnsiteOperatorSpL();
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         out += CreateOnsiteOperatorSpC(o);
      }
      return out;
   }
   
   CRS CreateOnsiteOperatorSm() const {
      CRS out = CreateOnsiteOperatorSmL();
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         out += CreateOnsiteOperatorSmC(o);
      }
      return out;
   }
   
   CRS CreateOnsiteOperatorSz() const {
      CRS out = CreateOnsiteOperatorSzL();
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         out += CreateOnsiteOperatorSzC(o);
      }
      return out;
   }
   
   CRS CreateOnsiteOperatorSx() const {
      CRS out = CreateOnsiteOperatorSxL();
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         out += CreateOnsiteOperatorSxC(o);
      }
      return out;
   }
   
   CRS CreateOnsiteOperatoriSy() const {
      CRS out = CreateOnsiteOperatoriSyL();
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         out += CreateOnsiteOperatoriSyC(o);
      }
      return out;
   }
   
private:
   const LatticeType lattice_;
   
   CQNType conserved_quantum_number_ = {std::vector<std::int32_t>{0}, 0};
   std::int32_t num_orbital_ = static_cast<std::int32_t>(conserved_quantum_number_.first.size());
   const std::int32_t dim_onsite_electron_ = 4;
   std::int32_t dim_onsite_all_electrons_ = static_cast<std::int32_t>(utility::CalculatePower(dim_onsite_electron_, num_orbital_));
   std::int32_t magnitude_2lspin_ = 1;
   std::int32_t dim_onsite_lspin_ = magnitude_2lspin_ + 1;
   std::int32_t dim_onsite_ = dim_onsite_all_electrons_*dim_onsite_lspin_;
   
   //! @brief The annihilation operator for the electrons
   //! with the orbital \f$ \alpha \f$ and the up spin, \f$ \hat{c}_{\alpha, \uparrow}\f$.
   std::vector<CRS> onsite_operator_c_up_;
   
   //! @brief The annihilation operator for the electrons
   //! with the orbital \f$ \alpha \f$ and the down spin \f$ \hat{c}_{\alpha, \downarrow}\f$.
   std::vector<CRS> onsite_operator_c_down_;
   
   //! @brief The creation operator for the electrons
   //! with the orbital \f$ \alpha \f$ and the up spin \f$ \hat{c}^{\dagger}_{\alpha, \uparrow}\f$.
   std::vector<CRS> onsite_operator_c_up_dagger_;
   
   //! @brief The the creation operator for the electrons
   //! with the orbital \f$ \alpha \f$ and the down spin \f$ \hat{c}^{\dagger}_{\alpha, \downarrow}\f$.
   std::vector<CRS> onsite_operator_c_down_dagger_;
   
   //! @brief The number operator for the electrons
   //! with the orbital \f$ \alpha \f$ and the up spin,
   //! \f$ \hat{n}_{\alpha, \uparrow}=\hat{c}^{\dagger}_{\alpha, \uparrow}\hat{c}_{\alpha, \uparrow}\f$.
   std::vector<CRS> onsite_operator_nc_up_;
   
   //! @brief The number operator for the electrons
   //! with the orbital \f$ \alpha \f$ and the down spin,
   //! \f$ \hat{n}_{\alpha, \downarrow}=\hat{c}^{\dagger}_{\alpha, \downarrow}\hat{c}_{\alpha, \downarrow}\f$.
   std::vector<CRS> onsite_operator_nc_down_;
   
   //! @brief The number operator for the electrons with the orbital
   //! \f$ \alpha \f$, \f$ \hat{n}_{\alpha}=\hat{n}_{\alpha, \uparrow} + \hat{n}_{\alpha, \downarrow}\f$.
   std::vector<CRS> onsite_operator_nc_;
   
   //! @brief The number operator for the electrons,
   //! \f$ \hat{n}=\sum_{\alpha}\left(\hat{n}_{\alpha, \uparrow} + \hat{n}_{\alpha, \downarrow}\right)\f$.
   CRS onsite_operator_nc_tot_;
   
   //! @brief The spin operator for the x-direction for the electrons with the orbital \f$ \alpha \f$,
   //! \f$ \hat{s}^{x}_{\alpha}=\frac{1}{2}(\hat{c}^{\dagger}_{\alpha, \uparrow}\hat{c}_{\alpha, \downarrow}
   //! + \hat{c}^{\dagger}_{\alpha, \downarrow}\hat{c}_{\alpha, \uparrow})\f$.
   std::vector<CRS> onsite_operator_sxc_;
   
   //! @brief The spin operator for the y-direction for the electrons with the orbital \f$ \alpha \f$,
   //! \f$ i\hat{s}^{y}_{\alpha}=\frac{1}{2}(\hat{c}^{\dagger}_{\alpha, \uparrow}\hat{c}_{\alpha, \downarrow}
   //! - \hat{c}^{\dagger}_{\alpha, \downarrow}\hat{c}_{\alpha, \uparrow})\f$.
   //! Here \f$ i=\sqrt{-1}\f$ is the the imaginary unit.
   std::vector<CRS> onsite_operator_isyc_;
   
   //! @brief The spin operator for the z-direction for the electrons with the orbital \f$ \alpha \f$,
   //! \f$ \hat{s}^{z}_{\alpha}=\frac{1}{2}(\hat{c}^{\dagger}_{\alpha, \uparrow}\hat{c}_{\alpha, \uparrow}
   //! - \hat{c}^{\dagger}_{\alpha, \downarrow}\hat{c}_{\alpha, \downarrow})\f$.
   std::vector<CRS> onsite_operator_szc_;
   
   //! @brief The raising operator for spin of the electrons with the orbital \f$ \alpha \f$,
   //! \f$ \hat{s}^{+}_{\alpha}=\hat{c}^{\dagger}_{\alpha, \uparrow}\hat{c}_{\alpha, \downarrow}\f$.
   std::vector<CRS> onsite_operator_spc_;
   
   //! @brief The lowering operator for spin of the electrons with the orbital \f$ \alpha \f$,
   //! \f$ \hat{s}^{-}_{\alpha}=\hat{c}^{\dagger}_{\alpha, \downarrow}\hat{c}_{\alpha, \uparrow}\f$.
   std::vector<CRS> onsite_operator_smc_;
   
   //! @brief The spin-\f$ S\f$ operator of the local spin for the x-direction \f$ \hat{S}^{x}\f$.
   CRS onsite_operator_sxl_;
   
   //! @brief The spin-\f$ S\f$ operator of the local spin for the y-direction \f$ i\hat{S}^{y}\f$ with \f$ i\f$ being
   //! the imaginary unit.
   CRS onsite_operator_isyl_;
   
   //! @brief The spin-\f$ S\f$ operator of the local spin for the z-direction \f$ \hat{S}^{z}\f$.
   CRS onsite_operator_szl_;
   
   //! @brief The spin-\f$ S\f$ raising operator of the local spin \f$ \hat{S}^{+}\f$.
   CRS onsite_operator_spl_;
   
   //! @brief The spin-\f$ S\f$ raising operator of the local spin \f$ \hat{S}^{-}\f$.
   CRS onsite_operator_sml_;
   
   //! @brief The correlation between the electron with the orbital \f$ \alpha \f$ spin and local spin
   //! \f$ \hat{\boldsymbol{s}}_{\alpha}\cdot\hat{\boldsymbol{S}}=
   //! \hat{s}^{x}_{\alpha}\hat{S}^{x}+\hat{s}^{y}_{\alpha}\hat{S}^{y}+\hat{s}^{z}_{\alpha}\hat{S}^{z}\f$
   std::vector<CRS> onsite_operator_scsl_;
   
   CRS onsite_operator_sp_;
   CRS onsite_operator_sm_;
   CRS onsite_operator_sz_;
   CRS onsite_operator_sx_;
   CRS onsite_operator_isy_;
   
   //! @brief Set onsite operators.
   void SetOnsiteOperator() {
      onsite_operator_c_up_.resize();
      onsite_operator_c_down_.resize();
      onsite_operator_c_up_dagger_.resize();
      onsite_operator_c_down_dagger_.resize();
      onsite_operator_nc_up_.resize();
      onsite_operator_nc_down_.resize();
      onsite_operator_nc_.resize();
      onsite_operator_sxc_.resize();
      onsite_operator_isyc_.resize();
      onsite_operator_szc_.resize();
      onsite_operator_spc_.resize();
      onsite_operator_smc_.resize();
      onsite_operator_scsl_.resize();
      for (std::int32_t o = 0; o < num_orbital_; ++o) {
         onsite_operator_c_up_[o] = CreateOnsiteOperatorCUp(o);
         onsite_operator_c_down_[o] = CreateOnsiteOperatorCDown(o);
         onsite_operator_c_up_dagger_[o] = CreateOnsiteOperatorCUpDagger(o);
         onsite_operator_c_down_dagger_[o] = CreateOnsiteOperatorCDownDagger(o);
         onsite_operator_nc_up_[o] = CreateOnsiteOperatorNCUp(o);
         onsite_operator_nc_down_[o] = CreateOnsiteOperatorNCDown(o);
         onsite_operator_nc_[o] = CreateOnsiteOperatorNC(o);
         onsite_operator_sxc_[o] = CreateOnsiteOperatorSxC(o);
         onsite_operator_isyc_[o] = CreateOnsiteOperatoriSyC(o);
         onsite_operator_szc_[o] = CreateOnsiteOperatorSzC(o);
         onsite_operator_spc_[o] = CreateOnsiteOperatorSpC(o);
         onsite_operator_smc_[o] = CreateOnsiteOperatorSmC(o);
         onsite_operator_scsl_[o] = CreateOnsiteOperatorSCSL(o);
      }
      onsite_operator_nc_tot_ = CreateOnsiteOperatorNCTot();
      onsite_operator_sxl_ = CreateOnsiteOperatorSxL();
      onsite_operator_isyl_ = CreateOnsiteOperatoriSyL();
      onsite_operator_szl_ = CreateOnsiteOperatorSzL();
      onsite_operator_spl_ = CreateOnsiteOperatorSpL();
      onsite_operator_sml_ = CreateOnsiteOperatorSmL();
      
      onsite_operator_sp_  = CreateOnsiteOperatorSp();
      onsite_operator_sm_  = CreateOnsiteOperatorSm();
      onsite_operator_sz_  = CreateOnsiteOperatorSz();
      onsite_operator_sx_  = CreateOnsiteOperatorSx();
      onsite_operator_isy_ = CreateOnsiteOperatoriSy();
   }
   
   //! @brief Calculate onsite basis for the electrons from an onsite basis.
   //! @param basis_onsite The onsite basis.
   //! @param orbital The electron orbital \f$ \alpha \f$.
   //! @return The onsite basis for the electrons.
   std::int32_t CalculateBasisOnsiteElectron(const std::int32_t basis_onsite, const std::int32_t orbital) const {
      const std::int32_t num_inner_electron = num_orbital_ - orbital - 1;
      std::int32_t basis_electron_onsite = basis_onsite/dim_onsite_lspin_;
      for (std::int32_t i = 0; i < num_inner_electron; ++i) {
         basis_electron_onsite /= dim_onsite_electron_;
      }
      return basis_electron_onsite%dim_onsite_electron_;
   }
   
   //! @brief Calculate the number of electrons from the onsite electron basis.
   //! @param basis_electron_onsite The onsite electron basis.
   //! @return The number of electrons
   std::int32_t CalculateNumElectronFromElectronBasis(const std::int32_t basis_electron_onsite) const {
      if (basis_electron_onsite == 0) {
         return 0;
      }
      else if (basis_electron_onsite == 1 || basis_electron_onsite == 2) {
         return 1;
      }
      else if (basis_electron_onsite == 3) {
         return 2;
      }
      else {
         std::stringstream ss;
         ss << "Unknown error detected in " << __FUNCTION__ << " at " << __LINE__ << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   std::int32_t CalculateBasisOnsiteLSpin(const std::int32_t basis_onsite) const {
      return basis_onsite%dim_onsite_lspin_;
   }
   
   static std::vector<std::vector<std::int32_t>> GenerateElectronConfigurations(const std::int32_t system_size,
                                                                                const std::int32_t total_electron) {
      const std::int32_t max_n_up_down = static_cast<std::int32_t>(total_electron/2);
      std::vector<std::vector<std::int32_t>> ele_configurations(4);
      for (std::int32_t n_up_down = 0; n_up_down <= max_n_up_down; ++n_up_down) {
         for (std::int32_t n_up = 0; n_up <= total_electron - 2*n_up_down; ++n_up) {
            const std::int32_t n_down = total_electron - 2*n_up_down - n_up;
            const std::int32_t n_vac = system_size - n_up - n_down - n_up_down;
            if (0 <= n_up && 0 <= n_down && 0 <= n_vac) {
               ele_configurations[0].push_back(n_up_down);
               ele_configurations[1].push_back(n_up);
               ele_configurations[2].push_back(n_down);
               ele_configurations[3].push_back(n_vac);
            }
         }
      }
      return ele_configurations;
   }
   
};


} // namespace quantum
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_QUANTUM_BASE_SPIN_MULTI_ELECTRON_HPP_ */
