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
//  mat_comp_hubbard.hpp
//  compnal
//
//  Created by kohei on 2023/01/18.
//  
//

#ifndef COMPNAL_SOLVER_ED_UTILITY_MAT_COMP_HUBBARD_HPP_
#define COMPNAL_SOLVER_ED_UTILITY_MAT_COMP_HUBBARD_HPP_

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
   const blas::CRS<RealType> &op_n = model.GetOnsiteOperatorNC();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t system_size = model.GetSystemSize();
   const std::vector<RealType> &hop_list = model.GetHoppingEnergy();
   const std::vector<RealType> &intersite_coulomb_list = model.GetIntersiteCoulomb();
   
   for (std::int32_t site = 0; site < system_size; ++site) {
      edmc->basis_onsite[site] = CalculateLocalBasis(basis, site, dim_onsite);
   }

   // Onsite elements
   for (std::int32_t site = 0; site < system_size; ++site) {
      GenerateMatrixComponentsOnsite(edmc, basis, site, onsite_ham);
   }

   // Intersite elements
   if (model.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
      // Hopping
      for (std::int32_t dist = 1; dist <= hop_list.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size; ++site) {
            std::int32_t num_electron = 0;
            std::int32_t f_sign = 1;
            for (std::int32_t i = site; i < site + dist; ++i) {
               num_electron += model.CalculateNumElectron(edmc->basis_onsite[i]);
            }
            if (num_electron%2 == 1) {
               f_sign = 1;
            }
            else {
               f_sign = -1;
            }
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up_d  , (site + dist)%system_size, op_c_up    , f_sign*hop_list[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up    , (site + dist)%system_size, op_c_up_d  , -f_sign*hop_list[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down_d, (site + dist)%system_size, op_c_down  , f_sign*hop_list[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down  , (site + dist)%system_size, op_c_down_d, -f_sign*hop_list[dist - 1]);
         }
      }
      
      // Intersite Coulomb
      for (std::int32_t dist = 1; dist <= intersite_coulomb_list.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size; ++site) {
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_n, (site + dist)%system_size, op_n, intersite_coulomb_list[dist - 1]);
         }
      }
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      // Hopping
      for (std::int32_t dist = 1; dist <= hop_list.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size - dist; ++site) {
            std::int32_t num_electron = 0;
            std::int32_t f_sign = 1;
            for (std::int32_t i = site; i < site + dist; ++i) {
               num_electron += model.CalculateNumElectron(edmc->basis_onsite[i]);
            }
            if (num_electron%2 == 1) {
               f_sign = 1;
            }
            else {
               f_sign = -1;
            }
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up_d  , (site + dist), op_c_up    , f_sign*hop_list[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up    , (site + dist), op_c_up_d  , -f_sign*hop_list[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down_d, (site + dist), op_c_down  , f_sign*hop_list[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down  , (site + dist), op_c_down_d, -f_sign*hop_list[dist - 1]);
         }
      }
      
      // Intersite Coulomb
      for (std::int32_t dist = 1; dist <= intersite_coulomb_list.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size - dist; ++site) {
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_n, (site + dist), op_n, intersite_coulomb_list[dist - 1]);
         }
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


template<typename RealType>
void GenerateMatrixComponents(ExactDiagMatrixComponents<RealType> *edmc,
                              const std::int64_t basis,
                              const model::quantum::Hubbard<lattice::Square, RealType> &model) {
   
   const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
   const blas::CRS<RealType> &op_c_up = model.GetOnsiteOperatorCUp();
   const blas::CRS<RealType> &op_c_up_d = model.GetOnsiteOperatorCUpDagger();
   const blas::CRS<RealType> &op_c_down = model.GetOnsiteOperatorCDown();
   const blas::CRS<RealType> &op_c_down_d = model.GetOnsiteOperatorCDownDagger();
   const blas::CRS<RealType> &op_n = model.GetOnsiteOperatorNC();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t x_size = model.GetLattice().GetXSize();
   const std::int32_t y_size = model.GetLattice().GetYSize();
   const std::vector<RealType> &hop_list = model.GetHoppingEnergy();
   const std::vector<RealType> &intersite_coulomb_list = model.GetIntersiteCoulomb();
   
   for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
      for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
         const std::int32_t index = coo_y*x_size + coo_x;
         edmc->basis_onsite[index] = CalculateLocalBasis(basis, index, dim_onsite);
      }
   }

   // Onsite elements
   for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
      for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
         const std::int32_t index = coo_y*x_size + coo_x;
         GenerateMatrixComponentsOnsite(edmc, basis, index, onsite_ham);
      }
   }

   // Intersite elements
   if (model.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
      // Hopping
      for (std::int32_t dist = 1; dist <= hop_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               const std::int32_t ind_x = coo_y*x_size + (coo_x + dist)%x_size;
               const std::int32_t ind_y = ((coo_y + dist)%y_size)*x_size + coo_x;
               std::int32_t num_electron_x = 0;
               std::int32_t num_electron_y = 0;
               std::int32_t f_sign = 1;
               for (std::int32_t i = ind; i < ind_x; ++i) {
                  num_electron_x += model.CalculateNumElectron(edmc->basis_onsite[i]);
               }
               
               if (num_electron_x%2 == 1) { f_sign = 1; }
               else { f_sign = -1;}
               
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_x, op_c_up    , f_sign*hop_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_x, op_c_up_d  , -f_sign*hop_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_x, op_c_down  , f_sign*hop_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_x, op_c_down_d, -f_sign*hop_list[dist - 1]);
               
               for (std::int32_t i = ind; i < ind_y; ++i) {
                  num_electron_y += model.CalculateNumElectron(edmc->basis_onsite[i]);
               }
               if (num_electron_y%2 == 1) { f_sign = 1; }
               else { f_sign = -1;}
               
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_y, op_c_up    , f_sign*hop_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_y, op_c_up_d  , -f_sign*hop_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_y, op_c_down  , f_sign*hop_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_y, op_c_down_d, -f_sign*hop_list[dist - 1]);
            }
         }
      }
      
      // Intersite Coulomb
      for (std::int32_t dist = 1; dist <= intersite_coulomb_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               const std::int32_t ind_x = coo_y*x_size + (coo_x + dist)%x_size;
               const std::int32_t ind_y = ((coo_y + dist)%y_size)*x_size + coo_x;
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_x, op_n, intersite_coulomb_list[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_y, op_n, intersite_coulomb_list[dist - 1]);
            }
         }
      }
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      // Hopping
      for (std::int32_t dist = 1; dist <= hop_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               if (coo_x + dist < x_size) {
                  const std::int32_t ind_x = coo_y*x_size + coo_x + dist;
                  std::int32_t num_electron_x = 0;
                  std::int32_t f_sign = 1;
                  for (std::int32_t i = ind; i < ind_x; ++i) {
                     num_electron_x += model.CalculateNumElectron(edmc->basis_onsite[i]);
                  }
                  if (num_electron_x%2 == 1) { f_sign = 1; }
                  else { f_sign = -1;}
                  
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_x, op_c_up    , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_x, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_x, op_c_down  , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_x, op_c_down_d, -f_sign*hop_list[dist - 1]);
               }
               if (coo_y + dist < y_size) {
                  const std::int32_t ind_y = (coo_y + dist)*x_size + coo_x;
                  std::int32_t num_electron_y = 0;
                  std::int32_t f_sign = 1;
                  for (std::int32_t i = ind; i < ind_y; ++i) {
                     num_electron_y += model.CalculateNumElectron(edmc->basis_onsite[i]);
                  }
                  if (num_electron_y%2 == 1) { f_sign = 1; }
                  else { f_sign = -1;}
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_y, op_c_up    , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_y, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_y, op_c_down  , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_y, op_c_down_d, -f_sign*hop_list[dist - 1]);
               }
            }
         }
      }
      
      // Intersite Coulomb
      for (std::int32_t dist = 1; dist <= intersite_coulomb_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               if (coo_x + dist < x_size) {
                  const std::int32_t ind_x = coo_y*x_size + coo_x + dist;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_x, op_n, intersite_coulomb_list[dist - 1]);
               }
               if (coo_y + dist < y_size) {
                  const std::int32_t ind_y = (coo_y + dist)*x_size + coo_x;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_y, op_n, intersite_coulomb_list[dist - 1]);
               }
            }
         }
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


template<typename RealType>
void GenerateMatrixComponents(ExactDiagMatrixComponents<RealType> *edmc,
                              const std::int64_t basis,
                              const model::quantum::Hubbard<lattice::Cubic, RealType> &model) {
   
   const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
   const blas::CRS<RealType> &op_c_up = model.GetOnsiteOperatorCUp();
   const blas::CRS<RealType> &op_c_up_d = model.GetOnsiteOperatorCUpDagger();
   const blas::CRS<RealType> &op_c_down = model.GetOnsiteOperatorCDown();
   const blas::CRS<RealType> &op_c_down_d = model.GetOnsiteOperatorCDownDagger();
   const blas::CRS<RealType> &op_n = model.GetOnsiteOperatorNC();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t x_size = model.GetLattice().GetXSize();
   const std::int32_t y_size = model.GetLattice().GetYSize();
   const std::int32_t z_size = model.GetLattice().GetZSize();
   const std::vector<RealType> &hop_list = model.GetHoppingEnergy();
   const std::vector<RealType> &intersite_coulomb_list = model.GetIntersiteCoulomb();
   
   for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
      for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
         for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
            const std::int32_t index = coo_z*x_size*y_size + coo_y*x_size + coo_x;
            edmc->basis_onsite[index] = CalculateLocalBasis(basis, index, dim_onsite);
         }
      }
   }

   // Onsite elements
   for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
      for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
         for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
            const std::int32_t index = coo_z*x_size*y_size + coo_y*x_size + coo_x;
            GenerateMatrixComponentsOnsite(edmc, basis, index, onsite_ham);
         }
      }
   }

   // Intersite elements
   if (model.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
      // Hopping
      for (std::int32_t dist = 1; dist <= hop_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind   = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + (coo_x + dist)%x_size;
                  const std::int32_t ind_y = coo_z*x_size*y_size + ((coo_y + dist)%y_size)*x_size + coo_x;
                  const std::int32_t ind_z = ((coo_z + dist)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  std::int32_t num_electron_x = 0;
                  std::int32_t num_electron_y = 0;
                  std::int32_t num_electron_z = 0;
                  std::int32_t f_sign = 1;
                  for (std::int32_t i = ind; i < ind_x; ++i) {
                     num_electron_x += model.CalculateNumElectron(edmc->basis_onsite[i]);
                  }
                  
                  if (num_electron_x%2 == 1) { f_sign = 1; }
                  else { f_sign = -1;}
                  
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_x, op_c_up    , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_x, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_x, op_c_down  , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_x, op_c_down_d, -f_sign*hop_list[dist - 1]);
                  
                  for (std::int32_t i = ind; i < ind_y; ++i) {
                     num_electron_y += model.CalculateNumElectron(edmc->basis_onsite[i]);
                  }
                  if (num_electron_y%2 == 1) { f_sign = 1; }
                  else { f_sign = -1;}
                  
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_y, op_c_up    , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_y, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_y, op_c_down  , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_y, op_c_down_d, -f_sign*hop_list[dist - 1]);
                  
                  for (std::int32_t i = ind; i < ind_z; ++i) {
                     num_electron_z += model.CalculateNumElectron(edmc->basis_onsite[i]);
                  }
                  if (num_electron_z%2 == 1) { f_sign = 1; }
                  else { f_sign = -1;}
                  
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_z, op_c_up    , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_z, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_z, op_c_down  , f_sign*hop_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_z, op_c_down_d, -f_sign*hop_list[dist - 1]);
               }
            }
         }
      }
      
      // Intersite Coulomb
      for (std::int32_t dist = 1; dist <= intersite_coulomb_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind   = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + (coo_x + dist)%x_size;
                  const std::int32_t ind_y = coo_z*x_size*y_size + ((coo_y + dist)%y_size)*x_size + coo_x;
                  const std::int32_t ind_z = ((coo_z + dist)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_x, op_n, intersite_coulomb_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_y, op_n, intersite_coulomb_list[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_z, op_n, intersite_coulomb_list[dist - 1]);
               }
            }
         }
      }
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      // Hopping
      for (std::int32_t dist = 1; dist <= hop_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind = coo_z*x_size*y_size + coo_y*x_size + coo_x;

                  if (coo_x + dist < x_size) {
                     const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + coo_x + dist;
                     std::int32_t num_electron_x = 0;
                     std::int32_t f_sign = 1;
                     for (std::int32_t i = ind; i < ind_x; ++i) {
                        num_electron_x += model.CalculateNumElectron(edmc->basis_onsite[i]);
                     }
                     
                     if (num_electron_x%2 == 1) { f_sign = 1; }
                     else { f_sign = -1;}
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_x, op_c_up    , f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_x, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_x, op_c_down  , f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_x, op_c_down_d, -f_sign*hop_list[dist - 1]);
                  }

                  if (coo_y + dist < y_size) {
                     const std::int32_t ind_y = coo_z*x_size*y_size + (coo_y + dist)*x_size + coo_x;
                     std::int32_t num_electron_y = 0;
                     std::int32_t f_sign = 1;
                     for (std::int32_t i = ind; i < ind_y; ++i) {
                        num_electron_y += model.CalculateNumElectron(edmc->basis_onsite[i]);
                     }
                     if (num_electron_y%2 == 1) { f_sign = 1; }
                     else { f_sign = -1;}
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_y, op_c_up    , f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_y, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_y, op_c_down  , f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_y, op_c_down_d, -f_sign*hop_list[dist - 1]);
                  }
                  
                  
                  
                  if (coo_z + dist < z_size) {
                     const std::int32_t ind_z = (coo_z + dist)*x_size*y_size + coo_y*x_size + coo_x;
                     std::int32_t num_electron_z = 0;
                     std::int32_t f_sign = 1;
                     for (std::int32_t i = ind; i < ind_z; ++i) {
                        num_electron_z += model.CalculateNumElectron(edmc->basis_onsite[i]);
                     }
                     if (num_electron_z%2 == 1) { f_sign = 1; }
                     else { f_sign = -1;}
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up_d  , ind_z, op_c_up    , f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_up    , ind_z, op_c_up_d  , -f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down_d, ind_z, op_c_down  , f_sign*hop_list[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_c_down  , ind_z, op_c_down_d, -f_sign*hop_list[dist - 1]);
                  }
               }
            }
         }
      }
      
      // Intersite Coulomb
      for (std::int32_t dist = 1; dist <= intersite_coulomb_list.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  if (coo_x + dist < x_size) {
                     const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + coo_x + dist;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_x, op_n, intersite_coulomb_list[dist - 1]);
                  }
                  if (coo_y + dist < y_size) {
                     const std::int32_t ind_y = coo_z*x_size*y_size + (coo_y + dist)*x_size + coo_x;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_y, op_n, intersite_coulomb_list[dist - 1]);
                  }
                  if (coo_z + dist < z_size) {
                     const std::int32_t ind_z = (coo_z + dist)*x_size*y_size + coo_y*x_size + coo_x;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_n, ind_z, op_n, intersite_coulomb_list[dist - 1]);
                  }
               }
            }
         }
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

#endif /* mat_comp_hubbaCOMPNAL_SOLVER_ED_UTILITY_MAT_COMP_HUBBARD_HPP_rd_chain_h */
