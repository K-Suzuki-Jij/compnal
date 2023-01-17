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
   
   void SetDiagonalizationMaxStep(const std::int32_t diag_max_step) {
      if (diag_max_step < 0) {
         throw std::runtime_error("diag_max_step must be larger than 1");
      }
      diag_max_step_ = diag_max_step;
   }
   
   void SetDiagonalizationAccuracy(const RealType diag_accuracy) {
      if (diag_accuracy < std::numeric_limits<RealType>::epsilon()) {
         throw std::runtime_error("Accuracy is too small");
      }
      diag_accuracy_ = diag_accuracy;
   }
   
   void SetLanczosStoreVector(const bool lanczos_store_vec) {
      lanczos_store_vec_ = lanczos_store_vec;
   }
   
   void SetInverseIterationMaxStep(const std::int32_t ii_max_step) {
      if (ii_max_step < 0) {
         throw std::runtime_error("The max step of the inverse iteration must be larger than 0");
      }
      ii_max_step_ = ii_max_step;
   }
   
   void SetInverseIterationShiftDiagElement(const RealType ii_diag_add) {
      ii_diag_add_ = ii_diag_add;
   }
   
   void SetInverseIterationAccuracy(const RealType ii_accuracy) {
      if (ii_accuracy < std::numeric_limits<RealType>::epsilon()) {
         throw std::runtime_error("ii_accuracy is too small");
      }
      ii_accuracy_ = ii_accuracy;
   }
   
   void SetConjugateGradientMaxStep(const std::int32_t linear_eq_max_step) {
      if (linear_eq_max_step < 0) {
         throw std::runtime_error("linear_eq_max_step must be larger than 0");
      }
      linear_eq_max_step_ = linear_eq_max_step;
   }
   
   void SetConjugateGradientAccuracy(const RealType linear_eq_accuracy) {
      if (linear_eq_accuracy < std::numeric_limits<RealType>::epsilon()) {
         throw std::runtime_error("linear_eq_accuracy must be is too small");
      }
      linear_eq_accuracy_ = linear_eq_accuracy;
   }
   
   void Diagonaliza(const std::int32_t level = 0) {
      if (level < 0) {
         throw std::runtime_error("The energy level must be non-negative integer");
      }
      
      const CQNType target_sector = model_.GetTaretSector();
      
      if (bases_.count(target_sector) == 0) {
         // Generate basis
         bases_[target_sector] = model_.GenerateBasis();
         inverse_bases_[target_sector] = GenerateInverseBasis(bases_.at(target_sector));
      }
      
      CRS ham = GenerateHamiltonian();

      if (eigenvalues_.size() < level + 1) {
         eigenvalues_.resize(level + 1);
         eigenvectors_.resize(level + 1);
      }
      
      if (diag_method_ == blas::DiagAlgorithm::LANCZOS) {
         blas::EigendecompositionLanczos(&eigenvalues_[0], &eigenvectors_[0], ham, GetDiagParams(), {});
      }
      else if (diag_method_ == blas::DiagAlgorithm::LOBPCG) {
         //blas::EigendecompositionLOBPCG(&eigenvalues_[0], &eigenvectors_[0], ham, diag_params);
      }
      else if (diag_method_ == blas::DiagAlgorithm::DAVIDSON) {
         //blas::EigendecompositionDavidson(&eigenvalues_[0], &eigenvectors_[0], ham, diag_params);
      }
      else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid diagonalization method detected." << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      blas::InverseIteration(&ham, &eigenvectors_[0], eigenvalues_[0], {}, GetIIParams());
   }
   
   
   
   
private:
   const ModelType model_;
   std::vector<RealType> eigenvalues_;
   std::vector<std::vector<RealType>> eigenvectors_;
   
   std::unordered_map<CQNType, std::vector<std::int64_t>, CQNHash> bases_;
   std::unordered_map<CQNType, std::unordered_map<std::int64_t, std::int64_t>, CQNHash> inverse_bases_;
   
   std::int32_t num_threads_ = 1;
   std::uint64_t seed_ = std::random_device()();
   
   blas::DiagAlgorithm diag_method_ = blas::DiagAlgorithm::LANCZOS;
   blas::LinearEqAlgorithm linear_eq_method_ = blas::LinearEqAlgorithm::CONJUGATE_GRADIENT;
   
   // Diagonalization
   std::int32_t diag_max_step_ = 1000;
   RealType diag_accuracy_ = std::pow(10, -14);
   bool lanczos_store_vec_ = false;
   
   //Inverse iteration
   std::int32_t ii_max_step_ = 3;
   RealType ii_diag_add_ = std::pow(10, -11);
   RealType ii_accuracy_ = std::pow(10, -7);
   
   //Conjugate Gradient
   std::int32_t linear_eq_max_step_ = 1000;
   RealType linear_eq_accuracy_ = std::pow(10, -7);
   
   blas::DiagParams<RealType> GetDiagParams() const {
      blas::DiagParams<RealType> diag_params;
      diag_params.flag_symmetric_crs = true;
      diag_params.num_threads = num_threads_;
      diag_params.max_step = diag_max_step_;
      diag_params.acc = diag_accuracy_;
      diag_params.flag_store_vec = lanczos_store_vec_;
      return diag_params;
   }
   
   blas::IIParams<RealType> GetIIParams() const {
      blas::IIParams<RealType> ii_params;
      ii_params.num_threads = num_threads_;
      ii_params.max_step = ii_max_step_;
      ii_params.diag_add = ii_diag_add_;
      ii_params.acc = ii_accuracy_;
      ii_params.linear_eq_params.num_threads = num_threads_;
      ii_params.linear_eq_params.flag_symmetric_crs = true;
      ii_params.linear_eq_params.flag_use_initial_vec = true;
      ii_params.linear_eq_method = linear_eq_method_;
      ii_params.linear_eq_params.max_step = linear_eq_max_step_;
      ii_params.linear_eq_params.acc = linear_eq_accuracy_;
      return ii_params;
   }
   
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
            
      std::vector<std::int64_t> num_row_element(dim_target + 1);
      
#pragma omp parallel num_threads(num_threads_)
      {
         ExactDiagMatrixComponents<RealType> components;
         components.site_constant.resize(model_.GetSystemSize());
         for (std::int32_t site = 0; site < model_.GetSystemSize(); ++site) {
            components.site_constant[site] =
            static_cast<std::int64_t>(std::pow(model_.GetDimOnsite(), site));
         }
         components.basis_onsite.resize(model_.GetSystemSize());
         
#pragma omp for schedule(guided)
         for (std::int64_t row = 0; row < dim_target; ++row) {
            GenerateMatrixComponents(&components, basis[row], model_);
            std::vector<std::int64_t> col_list;
            std::vector<RealType> val_list;
            const std::size_t size = components.basis_affected.size();
            for (std::size_t i = 0; i < size; ++i) {
               const std::int64_t a_basis = components.basis_affected[i];
               const RealType val = components.val[i];
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
            components.val.clear();
            components.basis_affected.clear();
            components.inv_basis_affected.clear();
         }
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
