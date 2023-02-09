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
//  base_spin_electron.hpp
//  compnal
//
//  Created by kohei on 2023/02/04.
//  
//

#ifndef COMPNAL_MODEL_QUANTUM_BASE_SPIN_ELECTRON_HPP_
#define COMPNAL_MODEL_QUANTUM_BASE_SPIN_ELECTRON_HPP_

#include "../../solver/ed_utility/ed_matrix_comp.hpp"
#include "../../lattice/all.hpp"
#include "../../blas/all.hpp"
#include "../../utility/hash.hpp"
#include "../../utility/integer.hpp"


#include <omp.h>
#include <vector>

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class BaseSpinElectron {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
   //! @brief Alias of compressed row strage (CRS) with RealType.
   using CRS = blas::CRS<RealType>;
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief The type of conserved quantum number (total electron, twice the number of total sz) pair.
   using CQNType = std::pair<std::int32_t, std::int32_t>;
   
   using CQNHash = utility::PairHash;
   
   BaseSpinElectron(const LatticeType &lattice): lattice_(lattice) {
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
   
   //! @brief Set the number of total electrons.
   //! @param total_electron The number of total electrons
   //! \f$ \hat{N}_{\rm e}=\sum^{N}_{i=1}\hat{n}_{i} \f$
   void SetTotalElectron(const std::int32_t total_electron) {
      if (total_electron < 0) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "total_electron must be a non-negative integer" << std::endl;
         throw std::runtime_error(ss.str());
      }
      conserved_quantum_number_.first = total_electron;
   }
   
   //! @brief Set the magnitude of the spin \f$ S \f$.
   //! @param magnitude_lspin The magnitude of the local spin \f$ S \f$.
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
         dim_onsite_ = dim_onsite_lspin_*dim_onsite_electron_;
         SetOnsiteOperator();
      }
   }
   
   //! @brief Calculate the number of electrons from the input onsite basis.
   //! @param basis_onsite The onsite basis.
   //! @return The number of electrons.
   std::int32_t CalculateNumElectron(const std::int32_t basis_onsite) const {
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      const std::int32_t basis_onsite_electron = basis_onsite/dim_onsite_lspin_;
      
      if (basis_onsite_electron == 0) {
         return 0;
      }
      else if (basis_onsite_electron == 1 || basis_onsite_electron == 2) {
         return 1;
      }
      else if (basis_onsite_electron == 3) {
         return 2;
      }
      else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid onsite basis" << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Print the onsite bases.
   void PrintBasisOnsite() const {
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         std::string b_ele = "None";
         if (CalculateBasisOnsiteElectron(row) == 0) {
            b_ele = "|vac>";
         } else if (CalculateBasisOnsiteElectron(row) == 1) {
            b_ele = "|↑>";
         } else if (CalculateBasisOnsiteElectron(row) == 2) {
            b_ele = "|↓>";
         } else if (CalculateBasisOnsiteElectron(row) == 3) {
            b_ele = "|↑↓>";
         }
         std::cout << "row " << row << ": " << b_ele << "|Sz=";
         std::cout << 0.5*magnitude_2lspin_ - CalculateBasisOnsiteLSpin(row) << ">" << std::endl;
      }
   }
   
   //! @brief Calculate sectors generated by an onsite operator.
   //! @tparam IntegerType Value type std::int32_t row and col.
   //! @param row The row in the matrix representation of an onsite operator.
   //! @param col The column in the matrix representation of an onsite operator.
   //! @return Sectors generated by an onsite operator.
   template<typename IntegerType>
   CQNType CalculateQNumber(const IntegerType row, const IntegerType col) const {
      static_assert(std::is_integral<IntegerType>::value, "Template parameter IntegerType must be integer type");
      if (row < 0 || col < 0 || dim_onsite_ <= row || dim_onsite_ <= col) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid parameters" << std::endl;
         throw std::runtime_error(ss.str());
      }
      // Calculate total sz from local spin.
      const std::int32_t row_lspin = row%dim_onsite_lspin_;
      const std::int32_t col_lspin = col%dim_onsite_lspin_;
      const std::int32_t total_sz_from_2lspin = 2*(col_lspin - row_lspin);
      
      // Calculate total electron and total sz from electrons.
      const std::int32_t row_electron = row/dim_onsite_lspin_;
      const std::int32_t col_electron = col/dim_onsite_lspin_;
      if (row_electron < 0 || col_electron < 0 || dim_onsite_electron_ <= row_electron ||
          dim_onsite_electron_ <= col_electron) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid parameters" << std::endl;
         throw std::runtime_error(ss.str());
      }
      if (row_electron == col_electron && 0 <= row_electron && row_electron < 4 && 0 <= col_electron &&
          col_electron < 4) {
         return {+0 + conserved_quantum_number_.first, +0 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 0 && col_electron == 1) {
         return {-1 + conserved_quantum_number_.first, -1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 0 && col_electron == 2) {
         return {-1 + conserved_quantum_number_.first, +1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 0 && col_electron == 3) {
         return {-2 + conserved_quantum_number_.first, +0 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 1 && col_electron == 0) {
         return {+1 + conserved_quantum_number_.first, +1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 1 && col_electron == 2) {
         return {+0 + conserved_quantum_number_.first, +2 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 1 && col_electron == 3) {
         return {-1 + conserved_quantum_number_.first, +1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 2 && col_electron == 0) {
         return {+1 + conserved_quantum_number_.first, -1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 2 && col_electron == 1) {
         return {+0 + conserved_quantum_number_.first, -2 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 2 && col_electron == 3) {
         return {-1 + conserved_quantum_number_.first, -1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 3 && col_electron == 0) {
         return {+2 + conserved_quantum_number_.first, +0 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 3 && col_electron == 1) {
         return {+1 + conserved_quantum_number_.first, -1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else if (row_electron == 3 && col_electron == 2) {
         return {+1 + conserved_quantum_number_.first, +1 + total_sz_from_2lspin + conserved_quantum_number_.second};
      } else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "The dimenstion of the matrix must be 4";
         throw std::runtime_error(ss.str());
      }
   }
   
   bool ValidateQNumber(const CQNType conserved_quantum_number) const {
      return ValidateQNumber(lattice_.GetSystemSize(), magnitude_2lspin_, conserved_quantum_number.first, conserved_quantum_number.second);
   }
   
   static bool ValidateQNumber(const std::int32_t system_size,
                               const std::int32_t magnitude_2lspin,
                               const std::int32_t total_electron,
                               const std::int32_t total_2sz) {
      if (system_size <= 0 || magnitude_2lspin <= 0 || total_electron < 0) {
         return false;
      }
      const bool c1 = (0 <= total_electron && total_electron <= 2 * system_size);
      const bool c2 = ((total_electron + system_size * magnitude_2lspin - total_2sz) % 2 == 0);
      const bool c3 = (-(total_electron + system_size * magnitude_2lspin) <= total_2sz);
      const bool c4 = (total_2sz <= total_electron + system_size * magnitude_2lspin);
      if (c1 && c2 && c3 && c4) {
         return true;
      } else {
         return false;
      }
   }
   
   std::int64_t CalculateTargetDim(const CQNType conserved_quantum_number) const {
      return CalculateTargetDim(lattice_.GetSystemSize(), magnitude_2lspin_, conserved_quantum_number.first, conserved_quantum_number.second);
   }
   
   static std::int64_t CalculateTargetDim(const std::int32_t system_size,
                                          const std::int32_t magnitude_2lspin,
                                          const std::int32_t total_electron,
                                          const std::int32_t total_2sz) {
      if (!ValidateQNumber(system_size, magnitude_2lspin, total_electron, total_2sz)) {
         return 0;
      }
      const std::vector<std::vector<std::int64_t>> binom = utility::GenerateBinomialTable(system_size);
      const std::int32_t max_n_up_down = static_cast<std::int32_t>(total_electron/2);
      std::int64_t dim = 0;
      for (std::int32_t n_up_down = 0; n_up_down <= max_n_up_down; ++n_up_down) {
         for (std::int32_t n_up = 0; n_up <= total_electron - 2*n_up_down; ++n_up) {
            const std::int32_t n_down = total_electron - 2*n_up_down - n_up;
            const std::int32_t n_vac = system_size - n_up - n_down - n_up_down;
            if (0 <= n_up && 0 <= n_down && 0 <= n_vac) {
               const std::int32_t total_2sz_electron = (n_up - n_down);
               const std::int64_t dim_electron = binom[system_size][n_up]*binom[system_size - n_up][n_down]*binom[system_size - n_up - n_down][n_up_down];
               const std::int64_t dim_lspin =
               BaseSpin<LatticeType, RealType>::CalculateTargetDim(system_size, magnitude_2lspin, total_2sz - total_2sz_electron);
               dim += dim_electron*dim_lspin;
            }
         }
      }
      return dim;
   }
   
   std::vector<std::int64_t> GenerateBasis(const CQNType conserved_quantum_number) const {
      
      if (!ValidateQNumber(conserved_quantum_number)) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid parameters (system_size or total_electron or total_sz)" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const std::int32_t total_electron = conserved_quantum_number.first;
      const std::int32_t total_2sz = conserved_quantum_number.second;
      const std::int32_t system_size = lattice_.GetSystemSize();
      
      std::vector<std::int64_t> site_constant_global(system_size);
      std::vector<std::int64_t> site_constant_electron(system_size);
      std::vector<std::int64_t> site_constant_lspin(system_size);
      for (std::int32_t site = 0; site < system_size; ++site) {
         site_constant_global[site] = utility::CalculatePower(dim_onsite_, site);
         site_constant_electron[site] = utility::CalculatePower(dim_onsite_electron_, site);
         site_constant_lspin[site] = utility::CalculatePower(dim_onsite_lspin_, site);
      }
      
      const std::int64_t dim_target_global = CalculateTargetDim(conserved_quantum_number);
      std::vector<std::int64_t> basis;
      basis.reserve(dim_target_global);
      
      const std::int32_t max_n_up_down = static_cast<std::int32_t>(total_electron/2);
      
      for (std::int32_t n_up_down = 0; n_up_down <= max_n_up_down; ++n_up_down) {
         for (std::int32_t n_up = 0; n_up <= total_electron - 2*n_up_down; ++n_up) {
            const std::int32_t n_down = total_electron - 2*n_up_down - n_up;
            const std::int32_t n_vac = system_size - n_up - n_down - n_up_down;
            if (0 <= n_up && 0 <= n_down && 0 <= n_vac) {
               const std::int32_t total_2sz_electron = n_up - n_down;
               const std::int32_t total_2sz_lspin = total_2sz - total_2sz_electron;
               if (BaseSpin<LatticeType, RealType>::ValidateQNumber(system_size, magnitude_2lspin_, total_2sz_lspin)) {
                  std::vector<std::int64_t> spin_basis =
                  BaseSpin<LatticeType, RealType>::GenerateBasis(total_2sz_lspin, system_size, magnitude_2lspin_, site_constant_global);
                  
                  std::vector<std::int32_t> integer_list(system_size);
                  for (std::int32_t s = 0; s < n_vac; ++s) {
                     integer_list[s] = 0;
                  }
                  for (std::int32_t s = 0; s < n_up; ++s) {
                     integer_list[s + n_vac] = 1;
                  }
                  for (std::int32_t s = 0; s < n_down; ++s) {
                     integer_list[s + n_vac + n_up] = 2;
                  }
                  for (std::int32_t s = 0; s < n_up_down; ++s) {
                     integer_list[s + n_vac + n_up + n_down] = 3;
                  }
                  std::sort(integer_list.begin(), integer_list.end());
                  
                  do {
                     std::int64_t electron_basis = 0;
                     for (std::size_t j = 0; j < integer_list.size(); ++j) {
                        electron_basis += integer_list[j]*site_constant_global[j];
                     }
                     for (std::size_t j = 0; j < spin_basis.size(); ++j) {
                        basis.push_back(spin_basis[j] + electron_basis*dim_onsite_lspin_);
                     }
                  } while (std::next_permutation(integer_list.begin(), integer_list.end()));
                  
               }
            }
         }
      }
      
      basis.shrink_to_fit();
      if (static_cast<std::int64_t>(basis.size()) != dim_target_global) {
         std::stringstream ss;
         ss << "Unknown error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "The size of basis is " << basis.size() << ", the dimension of the system is " << dim_target_global;
         throw std::runtime_error(ss.str());
      }
      
      std::sort(basis.begin(), basis.end());

      return basis;
   }
   
   CRS GenerateOnsiteOperatorCUp() const {
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.val.push_back(RealType{1.0});
         matrix.col.push_back(dim_onsite_lspin_ + element);
         matrix.row[element + 1] = matrix.col.size();
      }
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.row[element + 1 + dim_onsite_lspin_] = matrix.col.size();
      }
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.val.push_back(RealType{1.0});
         matrix.col.push_back(3*dim_onsite_lspin_ + element);
         matrix.row[element + 1 + 2*dim_onsite_lspin_] = matrix.col.size();
      }
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.row[element + 1 + 3*dim_onsite_lspin_] = matrix.col.size();
      }
      matrix.tag = blas::CRSTag::FERMION;
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatorCDown() const {
      //--------------------------------
      // # <->  [Cherge  ] -- (N,  2*sz)
      // 0 <->  [        ] -- (0,  0   )
      // 1 <->  [up      ] -- (1,  1   )
      // 2 <->  [down    ] -- (1, -1   )
      // 3 <->  [up&down ] -- (2,  0   )
      //--------------------------------
      
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.val.push_back(RealType{1.0});
         matrix.col.push_back(element + 2*dim_onsite_lspin_);
         matrix.row[element + 1] = matrix.col.size();
      }
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.val.push_back(-RealType{1.0});
         matrix.col.push_back(element + 3*dim_onsite_lspin_);
         matrix.row[element + 1 + dim_onsite_lspin_] = matrix.col.size();
      }
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.row[element + 1 + 2*dim_onsite_lspin_] = matrix.col.size();
      }
      for (std::int32_t element = 0; element < dim_onsite_lspin_; ++element) {
         matrix.row[element + 1 + 3*dim_onsite_lspin_] = matrix.col.size();
      }
      matrix.tag = blas::CRSTag::FERMION;
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatorSzL() const {
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row_c = 0; row_c < dim_onsite_electron_; ++row_c) {
         for (std::int32_t row_l = 0; row_l < dim_onsite_lspin_; ++row_l) {
            const RealType val = 0.5*magnitude_2lspin_ - row_l;
            if (val != 0.0) {
               matrix.val.push_back(val);
               matrix.col.push_back(row_l + row_c*dim_onsite_lspin_);
            }
            matrix.row[row_l + 1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         }
      }
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatorSpL() const {
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row_c = 0; row_c < dim_onsite_electron_; ++row_c) {
         matrix.row[row_c*dim_onsite_lspin_] = matrix.col.size();
         for (std::int32_t row_l = 1; row_l < dim_onsite_lspin_; ++row_l) {
            matrix.val.push_back(std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*2*row_l - row_l*(row_l + 1))));
            matrix.col.push_back(row_l + row_c*dim_onsite_lspin_);
            matrix.row[row_l + row_c*dim_onsite_lspin_] = matrix.col.size();
         }
      }
            
      matrix.row[dim_onsite_] = matrix.col.size();
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatorSmL() const {
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row_c = 0; row_c < dim_onsite_electron_; ++row_c) {
         matrix.row[1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         for (std::int32_t row_l = 1; row_l < dim_onsite_lspin_; ++row_l) {
            matrix.val.push_back(std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*2*row_l - row_l*(row_l + 1))));
            matrix.col.push_back(row_l - 1 + row_c*dim_onsite_lspin_);
            matrix.row[row_l + 1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         }
      }
      matrix.row[dim_onsite_] = matrix.col.size();
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatorSxL() const {
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row_c = 0; row_c < dim_onsite_electron_; ++row_c) {
         std::int32_t a = 0;
         std::int32_t b = 1;
         
         matrix.val.push_back(RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
         matrix.col.push_back(b + row_c*dim_onsite_lspin_);
         matrix.row[1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         
         for (std::int32_t row_l = 1; row_l < dim_onsite_lspin_ - 1; ++row_l) {
            a = row_l;
            b = row_l - 1;
            matrix.val.push_back(RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
            matrix.col.push_back(b + row_c*dim_onsite_lspin_);
            
            a = row_l;
            b = row_l + 1;
            matrix.val.push_back(RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
            matrix.col.push_back(b + row_c*dim_onsite_lspin_);
            matrix.row[row_l + 1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         }
         
         a = dim_onsite_lspin_ - 1;
         b = dim_onsite_lspin_ - 2;
         
         matrix.val.push_back(RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
         matrix.col.push_back(b + row_c*dim_onsite_lspin_);
         matrix.row[dim_onsite_lspin_ + row_c*dim_onsite_lspin_] = matrix.col.size();
      }
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatoriSyL() const {
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row_c = 0; row_c < dim_onsite_electron_; ++row_c) {
         std::int32_t a = 0;
         std::int32_t b = 1;
         
         matrix.val.push_back(RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
         matrix.col.push_back(b + row_c*dim_onsite_lspin_);
         matrix.row[1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         
         for (std::int32_t row_l = 1; row_l < dim_onsite_lspin_ - 1; ++row_l) {
            a = row_l;
            b = row_l - 1;
            
            matrix.val.push_back(-RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
            matrix.col.push_back(b + row_c*dim_onsite_lspin_);
            
            a = row_l;
            b = row_l + 1;
            matrix.val.push_back(RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
            matrix.col.push_back(b + row_c*dim_onsite_lspin_);
            
            matrix.row[row_l + 1 + row_c*dim_onsite_lspin_] = matrix.col.size();
         }
         
         a = dim_onsite_ - 1;
         b = dim_onsite_ - 2;
         
         matrix.val.push_back(-RealType{0.5}*std::sqrt(static_cast<RealType>((0.5*magnitude_2lspin_ + 1)*(a + b + 1) - (a + 1)*(b + 1))));
         matrix.col.push_back(b + row_c*dim_onsite_lspin_);
         matrix.row[dim_onsite_lspin_ + row_c*dim_onsite_lspin_] = matrix.col.size();
      }
      return matrix;
   }
   
   
   CRS GenerateOnsiteOperatorCUpDagger() const {
      return blas::GenerateTransposedMatrix(GenerateOnsiteOperatorCUp());
   }
   
   
   CRS GenerateOnsiteOperatorCDownDagger() const {
      return blas::GenerateTransposedMatrix(GenerateOnsiteOperatorCDown());
   }
   
   
   CRS GenerateOnsiteOperatorNCUp() const {
      return GenerateOnsiteOperatorCUpDagger()*GenerateOnsiteOperatorCUp();
   }
   
   
   CRS GenerateOnsiteOperatorNCDown() const {
      return GenerateOnsiteOperatorCDownDagger()*GenerateOnsiteOperatorCDown();
   }
   
   
   CRS GenerateOnsiteOperatorNC() const {
      return GenerateOnsiteOperatorNCUp() + GenerateOnsiteOperatorNCDown();
   }
   
   
   CRS GenerateOnsiteOperatorSxC() const {
      return RealType{0.5}*(GenerateOnsiteOperatorSpC() + GenerateOnsiteOperatorSmC());
   }
   
   
   CRS GenerateOnsiteOperatoriSyC() const {
      return RealType{0.5}*(GenerateOnsiteOperatorSpC() - GenerateOnsiteOperatorSmC());
   }
   
   
   CRS GenerateOnsiteOperatorSzC() const {
      return RealType{0.5}*(GenerateOnsiteOperatorNCUp() - GenerateOnsiteOperatorNCDown());
   }
   
   
   CRS GenerateOnsiteOperatorSpC() const {
      return GenerateOnsiteOperatorCUpDagger()*GenerateOnsiteOperatorCDown();
   }
   
   
   CRS GenerateOnsiteOperatorSmC() const {
      return GenerateOnsiteOperatorCDownDagger()*GenerateOnsiteOperatorCUp();
   }
   
   
   CRS GenerateOnsiteOperatorSCSL() const {
      const CRS spc = GenerateOnsiteOperatorSpC();
      const CRS smc = GenerateOnsiteOperatorSmC();
      const CRS szc = GenerateOnsiteOperatorSzC();
      const CRS spl = GenerateOnsiteOperatorSpL();
      const CRS sml = GenerateOnsiteOperatorSmL();
      const CRS szl = GenerateOnsiteOperatorSzL();
      return szc*szl + RealType{0.5}*(spc*sml + smc*spl);
   }
   
   CRS GenerateOnsiteOperatorSp() const {
      return GenerateOnsiteOperatorSpC() + GenerateOnsiteOperatorSpL();
   }
   
   CRS GenerateOnsiteOperatorSm() const {
      return GenerateOnsiteOperatorSmC() + GenerateOnsiteOperatorSmL();
   }
   
   CRS GenerateOnsiteOperatorSz() const {
      return GenerateOnsiteOperatorSzC() + GenerateOnsiteOperatorSzL();
   }
   
   CRS GenerateOnsiteOperatorSx() const {
      return GenerateOnsiteOperatorSxC() + GenerateOnsiteOperatorSxL();
   }
   
   CRS GenerateOnsiteOperatoriSy() const {
      return GenerateOnsiteOperatoriSyC() + GenerateOnsiteOperatoriSyL();
   }
   
   CQNType GetTaretSector() const {
      return conserved_quantum_number_;
   }
   
   //! @brief Get dimension of the local Hilbert space, \f$ 4*(2S+1)\f$.
   //! @return The dimension of the local Hilbert space, \f$ 4*(2S+1)\f$.
   std::int32_t GetDimOnsite() const { return dim_onsite_; }
   
   std::int32_t GetDimOnsiteELectron() const { return dim_onsite_electron_; }
   
   std::int32_t GetDimOnsiteLSpin() const { return dim_onsite_lspin_; }
   
   double GetTotalSz() const { return 0.5*conserved_quantum_number_.second; }
   
   double GetMagnitudeLSpin() const { return 0.5*magnitude_2lspin_; }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return lattice_.GetSystemSize();
   }
   
   //! @brief Get the boundary condition.
   //! @return The boundary condition.
   lattice::BoundaryCondition GetBoundaryCondition() const {
      return lattice_.GetBoundaryCondition();
   }
   
   //! @brief Get the lattice.
   //! @return The lattice.
   const LatticeType &GetLattice() const {
      return lattice_;
   }
   
   std::int32_t GetTotalElectron() const { return conserved_quantum_number_.first; }
   
   //! @brief Get the annihilation operator for the electrons with the up spin \f$ \hat{c}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{c}_{\uparrow}\f$.
   const CRS &GetOnsiteOperatorCUp() const { return onsite_operator_c_up_; }
   
   //! @brief Get the annihilation operator for the electrons with the down spin \f$ \hat{c}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{c}_{\downarrow}\f$.
   const CRS &GetOnsiteOperatorCDown() const { return onsite_operator_c_down_; }
   
   //! @brief Get the creation operator for the electrons with the up spin
   //! \f$ \hat{c}^{\dagger}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{c}^{\dagger}_{\uparrow}\f$.
   const CRS &GetOnsiteOperatorCUpDagger() const { return onsite_operator_c_up_dagger_; }
   
   //! @brief Get the creation operator for the electrons with the down spin
   //! \f$ \hat{c}^{\dagger}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{c}^{\dagger}_{\downarrow}\f$.
   const CRS &GetOnsiteOperatorCDownDagger() const { return onsite_operator_c_down_dagger_; }
   
   //! @brief Get the number operator for the electrons with the up spin
   //! \f$ \hat{n}_{\uparrow}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{n}_{\uparrow}\f$.
   const CRS &GetOnsiteOperatorNCUp() const { return onsite_operator_nc_up_; }
   
   //! @brief Get the number operator for the electrons with the down spin
   //! \f$ \hat{n}_{\downarrow}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{n}_{\downarrow}\f$.
   const CRS &GetOnsiteOperatorNCDown() const { return onsite_operator_nc_down_; }
   
   //! @brief Get the number operator for the electrons
   //! \f$ \hat{n}=\hat{n}_{\uparrow} + \hat{n}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{n}\f$.
   const CRS &GetOnsiteOperatorNC() const { return onsite_operator_nc_; }
   
   //! @brief Get the spin operator for the x-direction for the electrons
   //! \f$ \hat{s}^{x}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} +
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$.
   //! @return The matrix of \f$ \hat{s}^{x}\f$.
   const CRS &GetOnsiteOperatorSxC() const { return onsite_operator_sxc_; }
   
   //! @brief Get the spin operator for the y-direction for the electrons
   //! \f$ i\hat{s}^{y}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$. Here \f$ i=\sqrt{-1}\f$ is the the imaginary unit.
   //! @return The matrix of \f$ i\hat{s}^{y}\f$.
   const CRS &GetOnsiteOperatoriSyC() const { return onsite_operator_isyc_; }
   
   //! @brief Get the spin operator for the z-direction for the electrons
   //! \f$ \hat{s}^{z}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow})\f$.
   //! @return The matrix of \f$ \hat{s}^{z}\f$.
   const CRS &GetOnsiteOperatorSzC() const { return onsite_operator_szc_; }
   
   //! @brief Get the raising operator for spin of the electrons
   //! \f$ \hat{s}^{+}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow}\f$.
   //! @return The matrix of \f$ \hat{s}^{+}\f$.
   const CRS &GetOnsiteOperatorSpC() const { return onsite_operator_spc_; }
   
   //! @brief Get the lowering operator for spin of the electrons
   //! \f$ \hat{s}^{-}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow}\f$.
   //! @return The matrix of \f$ \hat{s}^{-}\f$.
   const CRS &GetOnsiteOperatorSmC() const { return onsite_operator_smc_; }
   
   //! @brief Get the spin-\f$ S\f$ operator of the local spin for the x-direction \f$ \hat{S}^{x}\f$.
   //! @return The matrix of \f$ \hat{S}^{x}\f$.
   const CRS &GetOnsiteOperatorSxL() const { return onsite_operator_sxl_; }
   
   //! @brief Get the spin-\f$ S\f$ operator of the local spin for the y-direction \f$ i\hat{S}^{y}\f$ with \f$ i\f$
   //! being the imaginary unit.
   //! @return The matrix of \f$ i\hat{S}^{y}\f$.
   const CRS &GetOnsiteOperatoriSyL() const { return onsite_operator_isyl_; }
   
   //! @brief Get the spin-\f$ S\f$ operator of the local spin for the z-direction \f$ \hat{S}^{z}\f$.
   //! @return The matrix of \f$ \hat{S}^{z}\f$.
   const CRS &GetOnsiteOperatorSzL() const { return onsite_operator_szl_; }
   
   //! @brief Get the spin-\f$ S\f$ raising operator of the local spin \f$ \hat{S}^{+}\f$.
   //! @return The matrix of \f$ \hat{S}^{+}\f$.
   const CRS &GetOnsiteOperatorSpL() const { return onsite_operator_spl_; }
   
   //! @brief Get the spin-\f$ S\f$ raising operator of the local spin \f$ \hat{S}^{-}\f$.
   //! @return The matrix of \f$ \hat{S}^{-}\f$.
   const CRS &GetOnsiteOperatorSmL() const { return onsite_operator_sml_; }
   
   //! @brief Get \f$
   //! \hat{\boldsymbol{s}}\cdot\hat{\boldsymbol{S}}=\hat{s}^{x}\hat{S}^{x}+\hat{s}^{y}\hat{S}^{y}+\hat{s}^{z}\hat{S}^{z}\f$
   //! @return The matrix of \f$ \hat{\boldsymbol{s}}\cdot\hat{\boldsymbol{S}}\f$.
   const CRS &GetOnsiteOperatorSCSL() const { return onsite_operator_scsl_; }
   
   const CRS &GetOnsiteOperatorSp() const { return onsite_operator_sp_; }
   const CRS &GetOnsiteOperatorSm() const { return onsite_operator_sm_; }
   const CRS &GetOnsiteOperatorSz() const { return onsite_operator_sz_; }
   const CRS &GetOnsiteOperatorSx() const { return onsite_operator_sx_; }
   const CRS &GetOnsiteOperatoriSy() const { return onsite_operator_isy_; }
   
   
private:
   const LatticeType lattice_;
   
   CQNType conserved_quantum_number_ = {0, 0};
   const std::int32_t dim_onsite_electron_ = 4;
   std::int32_t magnitude_2lspin_ = 1;
   std::int32_t dim_onsite_lspin_ = magnitude_2lspin_ + 1;
   std::int32_t dim_onsite_ = dim_onsite_electron_*dim_onsite_lspin_;
   
   //! @brief The annihilation operator for the electrons with the up spin \f$ \hat{c}_{\uparrow}\f$.
   CRS onsite_operator_c_up_;
   
   //! @brief The annihilation operator for the electrons with the down spin \f$ \hat{c}_{\downarrow}\f$.
   CRS onsite_operator_c_down_;
   
   //! @brief The creation operator for the electrons with the up spin \f$ \hat{c}^{\dagger}_{\uparrow}\f$.
   CRS onsite_operator_c_up_dagger_;
   
   //! @brief The the creation operator for the electrons with the down spin \f$ \hat{c}^{\dagger}_{\downarrow}\f$.
   CRS onsite_operator_c_down_dagger_;
   
   //! @brief The number operator for the electrons with the up spin \f$
   //! \hat{n}_{\uparrow}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow}\f$.
   CRS onsite_operator_nc_up_;
   
   //! @brief The number operator for the electrons with the down spin \f$
   //! \hat{n}_{\downarrow}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow}\f$.
   CRS onsite_operator_nc_down_;
   
   //! @brief The number operator for the electrons \f$ \hat{n}=\hat{n}_{\uparrow} + \hat{n}_{\downarrow}\f$.
   CRS onsite_operator_nc_;
   
   //! @brief The spin operator for the x-direction for the electrons
   //! \f$ \hat{s}^{x}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} +
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$.
   CRS onsite_operator_sxc_;
   
   //! @brief The spin operator for the y-direction for the electrons
   //! \f$ i\hat{s}^{y}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow})\f$. Here \f$ i=\sqrt{-1}\f$ is the the imaginary unit.
   CRS onsite_operator_isyc_;
   
   //! @brief The spin operator for the z-direction for the electrons
   //! \f$ \hat{s}^{z}=\frac{1}{2}(\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\uparrow} -
   //! \hat{c}^{\dagger}_{\downarrow}\hat{c}_{\downarrow})\f$.
   CRS onsite_operator_szc_;
   
   //! @brief The raising operator for spin of the electrons
   //! \f$ \hat{s}^{+}=\hat{c}^{\dagger}_{\uparrow}\hat{c}_{\downarrow}\f$.
   CRS onsite_operator_spc_;
   
   //! @brief The lowering operator for spin of the electrons
   //! \f$ \hat{s}^{-}=\hat{c}^{\dagger}_{\downarrow}\hat{c}_{\uparrow}\f$.
   CRS onsite_operator_smc_;
   
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
   
   //! @brief The correlation between the electron spin and local spin
   //! \f$ \hat{\boldsymbol{s}}\cdot\hat{\boldsymbol{S}}=\hat{s}^{x}\hat{S}^{x}+\hat{s}^{y}\hat{S}^{y}+\hat{s}^{z}\hat{S}^{z}\f$
   CRS onsite_operator_scsl_;
   
   CRS onsite_operator_sp_;
   CRS onsite_operator_sm_;
   CRS onsite_operator_sz_;
   CRS onsite_operator_sx_;
   CRS onsite_operator_isy_;
   
   
   void SetOnsiteOperator() {
      onsite_operator_c_up_ = GenerateOnsiteOperatorCUp();
      onsite_operator_c_down_ = GenerateOnsiteOperatorCDown();
      onsite_operator_c_up_dagger_ = GenerateOnsiteOperatorCUpDagger();
      onsite_operator_c_down_dagger_ = GenerateOnsiteOperatorCDownDagger();
      onsite_operator_nc_up_ = GenerateOnsiteOperatorNCUp();
      onsite_operator_nc_down_ = GenerateOnsiteOperatorNCDown();
      onsite_operator_nc_ = GenerateOnsiteOperatorNC();
      onsite_operator_sxc_ = GenerateOnsiteOperatorSxC();
      onsite_operator_isyc_ = GenerateOnsiteOperatoriSyC();
      onsite_operator_szc_ = GenerateOnsiteOperatorSzC();
      onsite_operator_spc_ = GenerateOnsiteOperatorSpC();
      onsite_operator_smc_ = GenerateOnsiteOperatorSmC();
      onsite_operator_sxl_ = GenerateOnsiteOperatorSxL();
      onsite_operator_isyl_ = GenerateOnsiteOperatoriSyL();
      onsite_operator_szl_ = GenerateOnsiteOperatorSzL();
      onsite_operator_spl_ = GenerateOnsiteOperatorSpL();
      onsite_operator_sml_ = GenerateOnsiteOperatorSmL();
      onsite_operator_scsl_ = GenerateOnsiteOperatorSCSL();
      onsite_operator_sp_ = GenerateOnsiteOperatorSp();
      onsite_operator_sm_ = GenerateOnsiteOperatorSm();
      onsite_operator_sz_ = GenerateOnsiteOperatorSz();
      onsite_operator_sx_ = GenerateOnsiteOperatorSx();
      onsite_operator_isy_ = GenerateOnsiteOperatoriSy();
   }
   
   //! @brief Calculate onsite basis for the electrons from an onsite basis.
   //! @param basis_onsite The onsite basis.
   //! @return The onsite basis for the electrons.
   std::int32_t CalculateBasisOnsiteElectron(const std::int32_t basis_onsite) const {
      return basis_onsite/dim_onsite_lspin_;
   }
   
   //! @brief Calculate onsite basis for the loca spins from an onsite basis.
   //! @param basis_onsite The onsite basis.
   //! @return The onsite basis for the local spins.
   std::int32_t CalculateBasisOnsiteLSpin(const std::int32_t basis_onsite) const {
      return basis_onsite%dim_onsite_lspin_;
   }
   
   
};


} // namespace quantum
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_QUANTUM_BASE_SPIN_ELECTRON_HPP_ */
