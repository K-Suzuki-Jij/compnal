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
//  mat_comp_hubbard_chain.hpp
//  compnal
//
//  Created by kohei on 2023/01/18.
//  
//

#ifndef COMPNAL_SOLVER_ED_UTILITY_MAT_COMP_HUBBARD_CHAIN_HPP_
#define COMPNAL_SOLVER_ED_UTILITY_MAT_COMP_HUBBARD_CHAIN_HPP_

#include "./ed_matrix_comp.hpp"

namespace compnal {
namespace solver {
namespace ed_utility {

template<typename RealType>
void GenerateMatrixComponents(ExactDiagMatrixComponents<RealType> *edmc,
                              const std::int64_t basis,
                              const model::quantum::Hubbard<lattice::Chain, RealType> &model) {
   
   const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
   const blas::CRS<RealType> &op_c_up = model.GetOnsiteOperatorCUp();
   const blas::CRS<RealType> &op_c_up_d = model.GetOnsiteOperatorCUpDagger();
   const blas::CRS<RealType> &op_c_down = model.GetOnsiteOperatorCDown();
   const blas::CRS<RealType> &op_c_down_d = model.GetOnsiteOperatorCDownDagger();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t system_size = model.GetSystemSize();
   const RealType hop = model.GetHoppingEnergy();
   std::int32_t fermion_sign = 1;
   
   for (std::int32_t site = 0; site < system_size; ++site) {
      edmc->basis_onsite[site] = CalculateLocalBasis(basis, site, dim_onsite);
   }

   // Onsite elements
   for (std::int32_t site = 0; site < system_size; ++site) {
      GenerateMatrixComponentsOnsite(edmc, basis, site, onsite_ham);
   }

   // Intersite elements
   if (model.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
      for (std::int32_t site = 0; site < system_size - 1; ++site) {
         if (model.CalculateNumElectron(edmc->basis_onsite[site])%2 == 1) {
            fermion_sign = 1;
         }
         else {
            fermion_sign = -1;
         }
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up_d  , site + 1, op_c_up    , fermion_sign*hop);
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up    , site + 1, op_c_up_d  , fermion_sign*hop);
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down_d, site + 1, op_c_down  , fermion_sign*hop);
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down  , site + 1, op_c_down_d, fermion_sign*hop);
      }
      GenerateMatrixComponentsIntersite(edmc, basis, 0, op_c_up_d  , system_size - 1, op_c_up    , fermion_sign*hop);
      GenerateMatrixComponentsIntersite(edmc, basis, 0, op_c_up    , system_size - 1, op_c_up_d  , fermion_sign*hop);
      GenerateMatrixComponentsIntersite(edmc, basis, 0, op_c_down_d, system_size - 1, op_c_down  , fermion_sign*hop);
      GenerateMatrixComponentsIntersite(edmc, basis, 0, op_c_down  , system_size - 1, op_c_down_d, fermion_sign*hop);
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      for (std::int32_t site = 0; site < system_size - 1; ++site) {
         if (model.CalculateNumElectron(edmc->basis_onsite[site])%2 == 1) {
            fermion_sign = 1;
         }
         else {
            fermion_sign = -1;
         }
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up_d  , site + 1, op_c_up    , fermion_sign*hop);
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up    , site + 1, op_c_up_d  , fermion_sign*hop);
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down_d, site + 1, op_c_down  , fermion_sign*hop);
         GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down  , site + 1, op_c_down_d, fermion_sign*hop);
      }
   }
   else {
      throw std::runtime_error("Unsupported BoundaryCondition");
   }
   
   // Fill zero in the diagonal elements for symmetric matrix vector product calculation.
   if (edmc->inv_basis_affected.count(basis) == 0) {
      edmc->inv_basis_affected[basis] = edmc->basis_affected.size();
      edmc->val.push_back(0.0);
      edmc->basis_affected.push_back(basis);
   }
}

} // namespace ed_utility
} // namespace solver
} // namespace compnal

#endif /* mat_comp_hubbaCOMPNAL_SOLVER_ED_UTILITY_MAT_COMP_HUBBARD_CHAIN_HPP_rd_chain_h */
