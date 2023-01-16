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
//  exact_diagonalization.hpp
//  compnal
//
//  Created by kohei on 2023/01/02.
//  
//

#ifndef COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_
#define COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_

#include "../blas/all.hpp"

#include <vector>
#include <omp.h>
#include <iostream>
#include <chrono>

namespace compnal {
namespace solver {

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

template<typename RealType>
struct ExactDiagLog {
  
   std::string execute_time;
   RealType time_gen_basis;
   RealType time_gen_ham;
   RealType time_diag;
   RealType time_ii;
   RealType time_cg;
   std::int32_t diag_step;
   std::int32_t ii_step;
   std::int32_t cg_step;
   
};

template<class ModelType>
class ExactDiag {
   using RealType = typename ModelType::ValueType;
   using CQNType = typename ModelType::CQNType;
   using CQNHash = typename ModelType::CQNHash;
   using CRS = blas::CRS<RealType>;
   
public:

   ExactDiag(const ModelType &model_): model_(model_) {}
   
   void SetNumThreads(const std::int32_t num_threads) {
      if (num_threads <= 0) {
         throw std::runtime_error("num_threads must be non-negative integer.");
      }
      num_threads_ = num_threads;
   }
   
   void Diagonaliza(const std::int32_t level = 0) {
      if (level < 0) {
         throw std::runtime_error("The energy level must be non-negative integer");
      }
      
      const CQNType target_sector = model_.GetTaretSector();
      
      if (bases_.count(target_sector) == 0) {
         // Generate basis
         printf("Gen basis...\n");
         bases_[target_sector] = model_.GenerateBasis();
         inverse_bases_[target_sector] = GenerateInverseBasis(bases_.at(target_sector));
      }
      
      printf("Gen ham...\n");
      CRS ham = GenerateHamiltonian();
      //ham.Print();
      //std::exit(1);
      printf("dim=%lld\n", ham.row_dim);
      if (eigenvalues_.size() < level + 1) {
         eigenvalues_.resize(level + 1);
         eigenvectors_.resize(level + 1);
      }
      
      if (diag_method == blas::DiagAlgorithm::LANCZOS) {
         printf("Diag ham...\n");
         diag_params.flag_symmetric_crs = true;
         diag_params.num_threads = num_threads_;
         auto start = std::chrono::system_clock::now();
         blas::EigendecompositionLanczos(&eigenvalues_[0], &eigenvectors_[0], ham, diag_params, {});
         std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
         std::cout << "Elapsed Time: " << elapsed_seconds.count() << " sec" << std::endl;
         std::exit(1);
      }
      else if (diag_method == blas::DiagAlgorithm::LOBPCG) {
         //blas::EigendecompositionLOBPCG(&eigenvalues_[0], &eigenvectors_[0], ham, diag_params);
      }
      else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid diagonalization method detected." << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      printf("II ham...\n");
      ii_params.cg.flag_symmetric_crs = true;
      ii_params.num_threads = num_threads_;
      ii_params.cg.num_threads = num_threads_;
      blas::InverseIteration(&ham, &eigenvectors_[0], eigenvalues_[0], {}, ii_params);
      
   }
   
   
   
   
private:
   const ModelType model_;
   std::vector<RealType> eigenvalues_;
   std::vector<std::vector<RealType>> eigenvectors_;
   
   std::unordered_map<CQNType, std::vector<std::int64_t>, CQNHash> bases_;
   std::unordered_map<CQNType, std::unordered_map<std::int64_t, std::int64_t>, CQNHash> inverse_bases_;
   
   std::int32_t num_threads_ = 1;
   
   //! @brief Diagonalization method.
   blas::DiagAlgorithm diag_method = blas::DiagAlgorithm::LANCZOS;
   blas::DiagParams<RealType> diag_params = blas::DiagParams<RealType>();
   blas::IIParams<RealType> ii_params = blas::IIParams<RealType>();

   std::unordered_map<std::int64_t, std::int64_t> GenerateInverseBasis(const std::vector<std::int64_t> &basis) const {
      const std::int64_t dim = static_cast<std::int64_t>(basis.size());
      std::unordered_map<std::int64_t, std::int64_t> inverse_basis;
      for (std::int64_t i = 0; i < dim; ++i) {
         inverse_basis[basis[i]] = i;
      }
      return inverse_basis;
   }
   
   std::int32_t CalculateLocalBasis(std::int64_t global_basis, const std::int32_t site, const std::int32_t dim_onsite) const {
      for (std::int32_t i = 0; i < site; ++i) {
         global_basis = global_basis/dim_onsite;
      }
      return static_cast<std::int32_t>(global_basis%dim_onsite);
   }
   
   CRS GenerateHamiltonian() const {
      auto start = std::chrono::system_clock::now();
      const CQNType target_sector = model_.GetTaretSector();
      const auto &basis = bases_.at(target_sector);
      const auto &basis_inv = inverse_bases_.at(target_sector);
      
      const std::int64_t dim_target = basis.size();
      std::int64_t num_total_elements = 0;
      
      std::vector<std::vector<std::int64_t>> temp_col(dim_target);
      std::vector<std::vector<RealType>> temp_val(dim_target);
      
      std::vector<ExactDiagMatrixComponents<RealType>> components(num_threads_);
      
      printf("A...\n");
      for (std::int32_t thread_num = 0; thread_num < num_threads_; ++thread_num) {
         components[thread_num].site_constant.resize(model_.GetSystemSize());
         for (std::int32_t site = 0; site < model_.GetSystemSize(); ++site) {
            components[thread_num].site_constant[site] =
            static_cast<std::int64_t>(std::pow(model_.GetDimOnsite(), site));
         }
         components[thread_num].basis_onsite.resize(model_.GetSystemSize());
      }
      printf("B...\n");
      std::vector<std::int64_t> num_row_element(dim_target + 1);
      
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int64_t row = 0; row < dim_target; ++row) {
         const std::int32_t thread_num = omp_get_thread_num();
         GenerateMatrixComponents(&components[thread_num], basis[row], model_);
         std::vector<std::int64_t> col_list;
         std::vector<RealType> val_list;
         const std::size_t size = components[thread_num].basis_affected.size();
         for (std::size_t i = 0; i < size; ++i) {
            const std::int64_t a_basis = components[thread_num].basis_affected[i];
            const RealType val = components[thread_num].val[i];
            if (basis_inv.count(a_basis) > 0) {
               const std::int64_t inv = basis_inv.at(a_basis);
               if ((inv < row && std::abs(val) > std::numeric_limits<RealType>::epsilon()) || inv == row) {
                  col_list.push_back(inv);
                  val_list.push_back(val);
                  num_row_element[row + 1]++;
               }
            }
            else if (basis_inv.count(a_basis) == 0 && std::abs(val) > std::numeric_limits<RealType>::epsilon()) {
               throw std::runtime_error("Matrix elements are not in the target space");
            }
         }
         temp_col[row] = col_list;
         temp_val[row] = val_list;
         components[thread_num].val.clear();
         components[thread_num].basis_affected.clear();
         components[thread_num].inv_basis_affected.clear();
      }
      
#pragma omp parallel for reduction(+: num_total_elements) num_threads(num_threads_)
      for (std::int64_t row = 0; row <= dim_target; ++row) {
         num_total_elements += num_row_element[row];
      }
      
      // Do not use openmp here
      for (std::int64_t row = 0; row < dim_target; ++row) {
         num_row_element[row + 1] += num_row_element[row];
      }
      
      CRS ham;
      ham.row.resize(dim_target + 1);
      ham.col.resize(num_total_elements);
      ham.val.resize(num_total_elements);
      
#pragma omp parallel for num_threads(num_threads_)
      for (std::int64_t row = 0; row < dim_target; ++row) {
         for (std::size_t i = 0; i < temp_col[row].size(); ++i) {
            ham.col[num_row_element[row]] = temp_col[row][i];
            ham.val[num_row_element[row]] = temp_val[row][i];
            num_row_element[row]++;
         }
         ham.row[row + 1] = num_row_element[row];
      }
      
      
      
      /*
      
      printf("C...\n");
#pragma omp parallel for reduction(+: num_total_elements) num_threads(num_threads_)
      for (std::int64_t row = 0; row <= dim_target; ++row) {
         num_total_elements += num_row_element[row];
      }
      
      // Do not use openmp here
      for (std::int64_t row = 0; row < dim_target; ++row) {
         num_row_element[row + 1] += num_row_element[row];
      }
      
      CRS ham;
      ham.row.resize(dim_target + 1);
      ham.col.resize(num_total_elements);
      ham.val.resize(num_total_elements);
      printf("D...\n");
#pragma omp parallel for schedule(guided) num_threads(num_threads_)
      for (std::int64_t row = 0; row < dim_target; ++row) {
         const std::int32_t thread_num = omp_get_thread_num();
         GenerateMatrixComponents(&components[thread_num], basis[row], model_);
         const std::size_t size = components[thread_num].basis_affected.size();
         for (std::size_t i = 0; i < size; ++i) {
            const std::int64_t a_basis = components[thread_num].basis_affected[i];
            const RealType val = components[thread_num].val[i];
            if (basis_inv.count(a_basis) > 0) {
               const std::int64_t inv = basis_inv.at(a_basis);
               if ((inv < row && std::abs(val) > std::numeric_limits<RealType>::epsilon()) || inv == row) {
                  ham.col[num_row_element[row]] = inv;
                  ham.val[num_row_element[row]] = val;
                  num_row_element[row]++;
               }
            }
         }
         ham.row[row + 1] = num_row_element[row];
         components[thread_num].val.clear();
         components[thread_num].basis_affected.clear();
         components[thread_num].inv_basis_affected.clear();
      }
       */
      printf("E...\n");
      ham.row_dim = dim_target;
      ham.col_dim = dim_target;
      
      const bool flag_check_1 = (ham.row[dim_target] != num_total_elements);
      const bool flag_check_2 = (static_cast<std::int64_t>(ham.col.size()) != num_total_elements);
      const bool flag_check_3 = (static_cast<std::int64_t>(ham.val.size()) != num_total_elements);
      
      if (flag_check_1 || flag_check_2 || flag_check_3) {
         std::stringstream ss;
         ss << "Unknown error detected in " << __FUNCTION__ << " at " << __LINE__ << std::endl;
         throw std::runtime_error(ss.str());
      }
      printf("F...\n");
      ham.SortCol();
         
      
      std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
      std::cout << "Elapsed Time: " << elapsed_seconds.count() << " sec" << std::endl;
      return ham;
   }
   
   void GenerateMatrixComponentsOnsite(ExactDiagMatrixComponents<RealType> *edmc,
                                       const std::int64_t basis,
                                       const std::int32_t site,
                                       const CRS &matrix_onsite) const {
      
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

   void GenerateMatrixComponentsIntersite(ExactDiagMatrixComponents<RealType> *edmc,
                                          const std::int64_t basis,
                                          const std::int32_t site_1,
                                          const CRS &matrix_onsite_1,
                                          const std::int32_t site_2,
                                          const CRS &matrix_onsite_2,
                                          const RealType coeef,
                                          const std::int32_t fermion_sign = 1.0) const {
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
            const std::int64_t a_basis = basis + (col_1 - basis_onsite_1) * site_constant_1 +
                                         (matrix_onsite_2.col[i2] - basis_onsite_2) * site_constant_2;
            if (a_basis <= basis) {
               if (edmc->inv_basis_affected.count(a_basis) == 0) {
                  edmc->inv_basis_affected[a_basis] = edmc->basis_affected.size();
                  edmc->val.push_back(fermion_sign*coeef*val_1*matrix_onsite_2.val[i2]);
                  edmc->basis_affected.push_back(a_basis);
               }
               else {
                  edmc->val[edmc->inv_basis_affected.at(a_basis)] +=
                      fermion_sign * coeef * val_1 * matrix_onsite_2.val[i2];
               }
            }
         }
      }
   }
   
   void GenerateMatrixComponents(ExactDiagMatrixComponents<RealType> *edmc,
                                 const std::int64_t basis,
                                 const model::quantum::Hubbard<lattice::Chain, RealType> &model) const {
      
      const blas::CRS<RealType> &onsite_ham = model.GenarateOnsiteOperatorHam();
      const blas::CRS<RealType> &op_c_up = model.GetOnsiteOperatorCUp();
      const blas::CRS<RealType> &op_c_up_d = model.GetOnsiteOperatorCUpDagger();
      const blas::CRS<RealType> &op_c_down = model.GetOnsiteOperatorCDown();
      const blas::CRS<RealType> &op_c_down_d = model.GetOnsiteOperatorCDownDagger();
      const std::int32_t dim_onsite = model.GetDimOnsite();
      const std::int32_t system_size = model.GetSystemSize();
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
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up_d  , site + 1, op_c_up    , fermion_sign*model.GetHoppingEnergy());
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up    , site + 1, op_c_up_d  , fermion_sign*model.GetHoppingEnergy());
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down_d, site + 1, op_c_down  , fermion_sign*model.GetHoppingEnergy());
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down  , site + 1, op_c_down_d, fermion_sign*model.GetHoppingEnergy());
         }
      }
      else if (model.GetBoundaryCondition() == lattice::BoundaryCondition::OBC) {
         for (std::int32_t site = 0; site < system_size - 1; ++site) {
            if (model.CalculateNumElectron(edmc->basis_onsite[site])%2 == 1) {
               fermion_sign = 1;
            }
            else {
               fermion_sign = -1;
            }
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up_d  , site + 1, op_c_up    , fermion_sign*model.GetHoppingEnergy());
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_up    , site + 1, op_c_up_d  , fermion_sign*model.GetHoppingEnergy());
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down_d, site + 1, op_c_down  , fermion_sign*model.GetHoppingEnergy());
            GenerateMatrixComponentsIntersite(edmc, basis, site, op_c_down  , site + 1, op_c_down_d, fermion_sign*model.GetHoppingEnergy());
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
   
   
};


} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_ */
