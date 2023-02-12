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
//  mat_comp_heisenberg.hpp
//  compnal
//
//  Created by kohei on 2023/01/22.
//  
//

#ifndef COMPNAL_SOLVER_UTILITY_EXACT_DIAG_MAT_COMP_HEISENBERG_HPP_
#define COMPNAL_SOLVER_UTILITY_EXACT_DIAG_MAT_COMP_HEISENBERG_HPP_

#include "./ed_matrix_comp.hpp"

namespace compnal {
namespace solver {
namespace utility_exact_diag {

template<typename RealType>
void GenerateMatrixComponents(ExactDiagMatrixComponents<RealType> *edmc,
                              const std::int64_t basis,
                              const model::quantum::Heisenberg<lattice::Chain, RealType> &model) {
   
   const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
   const blas::CRS<RealType> &op_sz = model.GetOnsiteOperatorSz();
   const blas::CRS<RealType> &op_sp = model.GetOnsiteOperatorSp();
   const blas::CRS<RealType> &op_sm = model.GetOnsiteOperatorSm();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t system_size = model.GetSystemSize();
   const std::vector<RealType> &spin_spin_z = model.GetSpinSpinZ();
   const std::vector<RealType> &spin_spin_xy = model.GetSpinSpinXY();
   
   for (std::int32_t site = 0; site < system_size; ++site) {
      edmc->basis_onsite[site] = CalculateLocalBasis(basis, site, dim_onsite);
   }

   // Onsite elements
   for (std::int32_t site = 0; site < system_size; ++site) {
      GenerateMatrixComponentsOnsite(edmc, basis, site, onsite_ham);
   }

   // Intersite elements
   if (model.GetBoundaryCondition() == lattice::BoundaryCondition::PBC) {
      // Spin-Spin-z
      for (std::int32_t dist = 1; dist <= spin_spin_z.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size; ++site) {
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_sz, (site + dist)%system_size, op_sz, spin_spin_z[dist - 1]);
         }
      }
      
      // Spin-Spin-xy
      for (std::int32_t dist = 1; dist <= spin_spin_xy.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size; ++site) {
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_sp, (site + dist)%system_size, op_sm, 0.5*spin_spin_xy[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_sm, (site + dist)%system_size, op_sp, 0.5*spin_spin_xy[dist - 1]);
         }
      }
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      // Spin-Spin-z
      for (std::int32_t dist = 1; dist <= spin_spin_z.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size - dist; ++site) {
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_sz, site + dist, op_sz, spin_spin_z[dist - 1]);
         }
      }
      
      // Spin-Spin-xy
      for (std::int32_t dist = 1; dist <= spin_spin_xy.size(); ++dist) {
         for (std::int32_t site = 0; site < system_size - dist; ++site) {
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_sp, site + dist, op_sm, 0.5*spin_spin_xy[dist - 1]);
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_sm, site + dist, op_sp, 0.5*spin_spin_xy[dist - 1]);
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
                              const model::quantum::Heisenberg<lattice::Square, RealType> &model) {
   
   const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
   const blas::CRS<RealType> &op_sz = model.GetOnsiteOperatorSz();
   const blas::CRS<RealType> &op_sp = model.GetOnsiteOperatorSp();
   const blas::CRS<RealType> &op_sm = model.GetOnsiteOperatorSm();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t x_size = model.GetLattice().GetXSize();
   const std::int32_t y_size = model.GetLattice().GetYSize();
   const std::vector<RealType> &spin_spin_z = model.GetSpinSpinZ();
   const std::vector<RealType> &spin_spin_xy = model.GetSpinSpinXY();
   
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
      // Spin-Spin-z
      for (std::int32_t dist = 1; dist <= spin_spin_z.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               const std::int32_t ind_x = coo_y*x_size + (coo_x + dist)%x_size;
               const std::int32_t ind_y = ((coo_y + dist)%y_size)*x_size + coo_x;
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_x, op_sz, spin_spin_z[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_y, op_sz, spin_spin_z[dist - 1]);
            }
         }
      }
      
      // Spin-Spin-xy
      for (std::int32_t dist = 1; dist <= spin_spin_xy.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               const std::int32_t ind_x = coo_y*x_size + (coo_x + dist)%x_size;
               const std::int32_t ind_y = ((coo_y + dist)%y_size)*x_size + coo_x;
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_x, op_sm, 0.5*spin_spin_xy[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_x, op_sp, 0.5*spin_spin_xy[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_y, op_sm, 0.5*spin_spin_xy[dist - 1]);
               GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_y, op_sp, 0.5*spin_spin_xy[dist - 1]);
            }
         }
      }
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      // Spin-Spin-z
      for (std::int32_t dist = 1; dist <= spin_spin_z.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               if (coo_x + dist < x_size) {
                  const std::int32_t ind_x = coo_y*x_size + coo_x + dist;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_x, op_sz, spin_spin_z[dist - 1]);
               }
               if (coo_y + dist < y_size) {
                  const std::int32_t ind_y = (coo_y + dist)*x_size + coo_x;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_y, op_sz, spin_spin_z[dist - 1]);
               }
            }
         }
      }
      
      // Spin-Spin-xy
      for (std::int32_t dist = 1; dist <= spin_spin_xy.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               const std::int32_t ind = coo_y*x_size + coo_x;
               if (coo_x + dist < x_size) {
                  const std::int32_t ind_x = coo_y*x_size + coo_x + dist;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_x, op_sm, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_x, op_sp, 0.5*spin_spin_xy[dist - 1]);
               }
               if (coo_y + dist < y_size) {
                  const std::int32_t ind_y = (coo_y + dist)*x_size + coo_x;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_y, op_sm, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_y, op_sp, 0.5*spin_spin_xy[dist - 1]);
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
                              const model::quantum::Heisenberg<lattice::Cubic, RealType> &model) {
   
   const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
   const blas::CRS<RealType> &op_sz = model.GetOnsiteOperatorSz();
   const blas::CRS<RealType> &op_sp = model.GetOnsiteOperatorSp();
   const blas::CRS<RealType> &op_sm = model.GetOnsiteOperatorSm();
   const std::int32_t dim_onsite = model.GetDimOnsite();
   const std::int32_t x_size = model.GetLattice().GetXSize();
   const std::int32_t y_size = model.GetLattice().GetYSize();
   const std::int32_t z_size = model.GetLattice().GetZSize();
   const std::vector<RealType> &spin_spin_z = model.GetSpinSpinZ();
   const std::vector<RealType> &spin_spin_xy = model.GetSpinSpinXY();
   
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
      // Spin-Spin-z
      for (std::int32_t dist = 1; dist <= spin_spin_z.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind   = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + (coo_x + dist)%x_size;
                  const std::int32_t ind_y = coo_z*x_size*y_size + ((coo_y + dist)%y_size)*x_size + coo_x;
                  const std::int32_t ind_z = ((coo_z + dist)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_x, op_sz, spin_spin_z[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_y, op_sz, spin_spin_z[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_z, op_sz, spin_spin_z[dist - 1]);
               }
            }
         }
      }
      
      // Spin-Spin-xy
      for (std::int32_t dist = 1; dist <= spin_spin_xy.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind   = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + (coo_x + dist)%x_size;
                  const std::int32_t ind_y = coo_z*x_size*y_size + ((coo_y + dist)%y_size)*x_size + coo_x;
                  const std::int32_t ind_z = ((coo_z + dist)%z_size)*x_size*y_size + coo_y*x_size + coo_x;
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_x, op_sm, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_x, op_sp, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_y, op_sm, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_y, op_sp, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_z, op_sm, 0.5*spin_spin_xy[dist - 1]);
                  GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_z, op_sp, 0.5*spin_spin_xy[dist - 1]);
               }
            }
         }
      }
   }
   else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
      // Spin-Spin-z
      for (std::int32_t dist = 1; dist <= spin_spin_z.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  if (coo_x + dist < x_size) {
                     const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + coo_x + dist;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_x, op_sz, spin_spin_z[dist - 1]);
                  }
                  if (coo_y + dist < y_size) {
                     const std::int32_t ind_y = coo_z*x_size*y_size + (coo_y + dist)*x_size + coo_x;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_y, op_sz, spin_spin_z[dist - 1]);
                  }
                  if (coo_z + dist < z_size) {
                     const std::int32_t ind_z = (coo_z + dist)*x_size*y_size + coo_y*x_size + coo_x;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sz, ind_z, op_sz, spin_spin_z[dist - 1]);
                  }
               }
            }
         }
      }
      
      // Spin-Spin-xy
      for (std::int32_t dist = 1; dist <= spin_spin_xy.size(); ++dist) {
         for (std::int32_t coo_x = 0; coo_x < x_size; ++coo_x) {
            for (std::int32_t coo_y = 0; coo_y < y_size; ++coo_y) {
               for (std::int32_t coo_z = 0; coo_z < z_size; ++coo_z) {
                  const std::int32_t ind   = coo_z*x_size*y_size + coo_y*x_size + coo_x;
                  if (coo_x + dist < x_size) {
                     const std::int32_t ind_x = coo_z*x_size*y_size + coo_y*x_size + coo_x + dist;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_x, op_sm, 0.5*spin_spin_xy[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_x, op_sp, 0.5*spin_spin_xy[dist - 1]);
                  }
                  if (coo_y + dist < y_size) {
                     const std::int32_t ind_y = coo_z*x_size*y_size + (coo_y + dist)*x_size + coo_x;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_y, op_sm, 0.5*spin_spin_xy[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_y, op_sp, 0.5*spin_spin_xy[dist - 1]);
                  }
                  if (coo_z + dist < z_size) {
                     const std::int32_t ind_z = (coo_z + dist)*x_size*y_size + coo_y*x_size + coo_x;
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sp, ind_z, op_sm, 0.5*spin_spin_xy[dist - 1]);
                     GenerateMatrixComponentsIntersite(edmc, basis, ind, op_sm, ind_z, op_sp, 0.5*spin_spin_xy[dist - 1]);
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




} // namespace utility_exact_diag
} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_UTILITY_EXACT_DIAG_MAT_COMP_HEISENBERG_HPP_ */
