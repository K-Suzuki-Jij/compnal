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
//  ed_matrix_comp.hpp
//  compnal
//
//  Created by kohei on 2023/01/18.
//  
//

#ifndef COMPNAL_SOLVER_UTILITY_EXACT_DIAG_ED_MATRIX_COMP_HPP_
#define COMPNAL_SOLVER_UTILITY_EXACT_DIAG_ED_MATRIX_COMP_HPP_

#include "../../blas/all.hpp"

namespace compnal {
namespace solver {
namespace utility_exact_diag {

//! @brief Information for calculating the matrix elements of the Hamiltonian.
template<typename RealType>
struct ExactDiagMatrixComponents {
   //! @brief Values of the matrix elements.
   std::vector<RealType> val;

   //! @brief Column number of the matrix elements.
   std::vector<std::int64_t> basis_affected;

   //! @brief The onsite basis.
   std::vector<std::int32_t> basis_onsite;

   //! @brief Constants for calculating matrix elements.
   std::vector<std::int64_t> site_constant;

   //! @brief Inverse basis.
   std::unordered_map<std::int64_t, std::int64_t> inv_basis_affected;
   
};

std::int32_t CalculateLocalBasis(std::int64_t global_basis,
                                 const std::int32_t site,
                                 const std::int32_t dim_onsite) {
   for (std::int32_t i = 0; i < site; ++i) {
      global_basis = global_basis/dim_onsite;
   }
   return static_cast<std::int32_t>(global_basis%dim_onsite);
}

template<typename RealType>
void GenerateMatrixComponentsOnsite(ExactDiagMatrixComponents<RealType> *edmc,
                                    const std::int64_t basis,
                                    const std::int32_t site,
                                    const blas::CRS<RealType> &matrix_onsite) {
   
   const std::int32_t basis_onsite = edmc->basis_onsite[site];
   const std::int64_t site_constant = edmc->site_constant[site];

   for (std::int64_t i = matrix_onsite.row[basis_onsite]; i < matrix_onsite.row[basis_onsite + 1]; ++i) {
      const std::int64_t a_basis = basis + (matrix_onsite.col[i] - basis_onsite)*site_constant;
      if (a_basis <= basis) {
         if (edmc->inv_basis_affected.count(a_basis) == 0) {
            edmc->inv_basis_affected[a_basis] = edmc->basis_affected.size();
            edmc->val.push_back(matrix_onsite.val[i]);
            edmc->basis_affected.push_back(a_basis);
         }
         else {
            edmc->val[edmc->inv_basis_affected.at(a_basis)] += matrix_onsite.val[i];
         }
      }
   }
}

template<typename RealType>
void GenerateMatrixComponentsIntersite(ExactDiagMatrixComponents<RealType> *edmc,
                                       const std::int64_t basis,
                                       const std::int32_t site_1,
                                       const blas::CRS<RealType> &matrix_onsite_1,
                                       const std::int32_t site_2,
                                       const blas::CRS<RealType> &matrix_onsite_2,
                                       const RealType coeef) {
   
   if (std::abs(coeef) <= std::numeric_limits<RealType>::epsilon()) {
      return;
   }

   const std::int32_t basis_onsite_1 = edmc->basis_onsite[site_1];
   const std::int32_t basis_onsite_2 = edmc->basis_onsite[site_2];
   const std::int64_t site_constant_1 = edmc->site_constant[site_1];
   const std::int64_t site_constant_2 = edmc->site_constant[site_2];

   for (std::int64_t i1 = matrix_onsite_1.row[basis_onsite_1]; i1 < matrix_onsite_1.row[basis_onsite_1 + 1]; ++i1) {
      const RealType val_1 = matrix_onsite_1.val[i1];
      const std::int64_t col_1 = matrix_onsite_1.col[i1];
      for (std::int64_t i2 = matrix_onsite_2.row[basis_onsite_2]; i2 < matrix_onsite_2.row[basis_onsite_2 + 1];
           ++i2) {
         const std::int64_t a_basis = basis + (col_1 - basis_onsite_1)*site_constant_1 +
                                      (matrix_onsite_2.col[i2] - basis_onsite_2)*site_constant_2;
         if (a_basis <= basis) {
            if (edmc->inv_basis_affected.count(a_basis) == 0) {
               edmc->inv_basis_affected[a_basis] = edmc->basis_affected.size();
               edmc->val.push_back(coeef*val_1*matrix_onsite_2.val[i2]);
               edmc->basis_affected.push_back(a_basis);
            }
            else {
               edmc->val[edmc->inv_basis_affected.at(a_basis)] += coeef*val_1*matrix_onsite_2.val[i2];
            }
         }
      }
   }
}

} // namespace utility_exact_diag
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_UTILITY_EXACT_DIAG_ED_MATRIX_COMP_HPP_ */
