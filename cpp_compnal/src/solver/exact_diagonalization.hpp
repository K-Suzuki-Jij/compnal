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
#include "./ed_utility/all.hpp"

#include <vector>
#include <omp.h>
#include <iostream>
#include <chrono>

namespace compnal {
namespace solver {

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
   
   RealType GetEigenvalue(const std::int32_t level) const {
      return eigenvalues_.at(level);
   }
   
   const std::vector<RealType> &GetEigenvector(const std::int32_t level) const {
      return eigenvectors_.at(level);
   }
   
   const std::vector<RealType> &GetEigenvalues() const {
      return eigenvalues_;
   }
   
   const std::vector<std::vector<RealType>> &GetEigenvectors() const {
      return eigenvectors_;
   }
   
   void CalculateGroundState() {
      CalculateTargetState(0);
   }
   
   void CalculateTargetState(const std::int32_t level) {
      if (level < 0) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "Invalid target_sector: " << level << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const CQNType target_sector = model_.GetTaretSector();
      
      if (bases_.count(target_sector) == 0) {
         const auto start = std::chrono::system_clock::now();
         bases_[target_sector] = model_.GenerateBasis(target_sector, num_threads_);
         inverse_bases_[target_sector] = GenerateInverseBasis(bases_.at(target_sector));
         if (flag_display_info_) {
            const std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
            std::cout << "\rGenerate Basis: " << elapsed_seconds.count() << " [sec]" << std::endl;
            std::cout << std::flush;
         }
      }
      
      auto ham = GenerateHamiltonian();
      for (std::int32_t i = 0; i <= level; ++i) {
         if (eigenvalues_.size() > i) {
            continue;
         }
         
         RealType temp_eigenvalue = 0;
         std::vector<RealType> temp_eigenvector(bases_.at(target_sector).size());
         if (diag_method_ == blas::DiagAlgorithm::LANCZOS) {
            blas::EigendecompositionLanczos(&temp_eigenvalue, &temp_eigenvector, ham, eigenvectors_, GetDiagParams());
         }
         else if (diag_method_ == blas::DiagAlgorithm::LOBPCG) {
            throw std::runtime_error("LOBPCG is under construction");
         }
         else if (diag_method_ == blas::DiagAlgorithm::DAVIDSON) {
            throw std::runtime_error("DAVIDSON is under construction");
         }
         else {
            std::stringstream ss;
            ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
            ss << "Invalid diagonalization method detected." << std::endl;
            throw std::runtime_error(ss.str());
         }
         blas::InverseIteration(&ham, &temp_eigenvector, temp_eigenvalue, eigenvectors_, GetIIParams());
         eigenvalues_.push_back(temp_eigenvalue);
         eigenvectors_.push_back(temp_eigenvector);
      }
   }
   
   RealType CalculateExpectationValue(const CRS &m,
                                      const typename ModelType::COOIndexType site_index,
                                      const std::int32_t level = 0) const {
      if (eigenvectors_.size() <= level) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "An eigenvector of the energy level: " << level << " has not been calculated" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (m.row_dim != model_.GetDimOnsite()) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The dimension of local matrix is not equal to the onsite dimension of the model" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (!model_.GetLattice().ValidateCOOIndex(site_index)) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The input site is invalid" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (m.tag == blas::CRSTag::FERMION) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The expectation value of Ferimon operators cannot be calculated " << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const CQNType target_sector = model_.GetTaretSector();
      const auto &basis = bases_.at(target_sector);
      const auto &basis_inv = inverse_bases_.at(target_sector);
      const auto &eigenvector = eigenvectors_.at(level);
      
      const std::int64_t dim = static_cast<std::int64_t>(eigenvector.size());
      const std::int32_t dim_onsite = static_cast<std::int32_t>(m.row_dim);
      const std::int32_t one_dim_site_index = model_.GetLattice().CalculateOneDimSiteIndex(site_index);
      const std::int64_t site_constant = static_cast<std::int64_t>(std::pow(dim_onsite, one_dim_site_index));
      
      RealType val = 0.0;
      
#pragma omp parallel for schedule(guided) reduction(+ : val) num_threads(num_threads_)
      for (std::int64_t i = 0; i < dim; ++i) {
         const std::int64_t global_basis = basis[i];
         const std::int32_t local_basis = ed_utility::CalculateLocalBasis(global_basis, one_dim_site_index, dim_onsite);
         RealType temp_val = 0.0;
         for (std::int64_t j = m.row[local_basis]; j < m.row[local_basis + 1]; ++j) {
            const std::int64_t a_basis = global_basis - (local_basis - m.col[j])*site_constant;
            if (basis_inv.count(a_basis) != 0) {
               temp_val += eigenvector[basis_inv.at(a_basis)]*m.val[j];
            }
         }
         val += eigenvector[i]*temp_val;
      }
      return val;
   }
   
   RealType CalculateCorrelationFunction(const CRS &m_1,
                                         const typename ModelType::COOIndexType site_index_1,
                                         const CRS &m_2,
                                         const typename ModelType::COOIndexType site_index_2,
                                         const std::int32_t level = 0) {
      
      if (eigenvectors_.size() <= level) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "An eigenvector of the energy level: " << level << " has not been calculated" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (m_1.row_dim != m_2.row_dim || m_1.row_dim != m_1.col_dim || m_2.row_dim != m_2.col_dim) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "Invalid input of the local operators" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (m_1.row_dim != model_.GetDimOnsite()) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The dimension of local matrix is not equal to the onsite dimension of the model" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (!model_.GetLattice().ValidateCOOIndex(site_index_1) || !model_.GetLattice().ValidateCOOIndex(site_index_2)) {
         std::stringstream ss;
         ss << "Error in " << __func__ << std::endl;
         ss << "The input site is invalid" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const CRS m1_dagger = blas::GenerateTransposedMatrix(m_1);
      const auto target_sector_set = GenerateTargetSector(m1_dagger, m_2);
      const auto &basis_inv = inverse_bases_.at(model_.GetTaretSector());
      const std::int32_t dim_onsite = model_.GetDimOnsite();
      const std::int32_t one_dim_site_index_1 = model_.GetLattice().CalculateOneDimSiteIndex(site_index_1);
      const std::int32_t one_dim_site_index_2 = model_.GetLattice().CalculateOneDimSiteIndex(site_index_2);
      const std::int64_t site_constant_m1 = static_cast<std::int64_t>(std::pow(dim_onsite, one_dim_site_index_1));
      const std::int64_t site_constant_m2 = static_cast<std::int64_t>(std::pow(dim_onsite, one_dim_site_index_2));
      const auto &eigenvector = eigenvectors_.at(level);
      std::vector<RealType> vector_work_m1;
      std::vector<RealType> vector_work_m2;
      RealType val = 0.0;
      
      for (const auto &sector : target_sector_set) {
         if (bases_.count(sector) == 0) {
            const auto start = std::chrono::system_clock::now();
            bases_[sector] = model_.GenerateBasis(sector, num_threads_);
            inverse_bases_[sector] = GenerateInverseBasis(bases_.at(sector));
            if (flag_display_info_) {
               const std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
               std::cout << "\rGenerate Basis: " << elapsed_seconds.count() << " [sec]" << std::endl;
               std::cout << std::flush;
            }
         }
                  
         const auto &basis = bases_.at(sector);
         const std::int64_t dim_target = static_cast<std::int64_t>(basis.size());
         vector_work_m1.resize(dim_target);
         vector_work_m2.resize(dim_target);
         
#pragma omp parallel for num_threads(num_threads_)
         for (std::int64_t i = 0; i < dim_target; ++i) {
            const std::int64_t global_basis = basis[i];
            const std::int32_t local_basis_m1 = ed_utility::CalculateLocalBasis(global_basis,
                                                                                one_dim_site_index_1,
                                                                                dim_onsite);
            const std::int32_t local_basis_m2 = ed_utility::CalculateLocalBasis(global_basis,
                                                                                one_dim_site_index_2,
                                                                                dim_onsite);
            RealType temp_val_m1 = 0.0;
            RealType temp_val_m2 = 0.0;
            
            std::int32_t fermion_sign_m1 = 1;
            if (m_1.tag == blas::CRSTag::FERMION) {
               std::int32_t num_electron = 0;
               for (std::int32_t site = 0; site < one_dim_site_index_1; site++) {
                  num_electron += model_.CalculateNumElectron(ed_utility::CalculateLocalBasis(global_basis, site, dim_onsite));
               }
               if (num_electron%2 == 1) {
                  fermion_sign_m1 = -1;
               }
            }
            
            for (std::int64_t j = m1_dagger.row[local_basis_m1]; j < m1_dagger.row[local_basis_m1 + 1]; ++j) {
               const std::int64_t a_basis = global_basis - (local_basis_m1 - m1_dagger.col[j]) * site_constant_m1;
               if (basis_inv.count(a_basis) != 0) {
                  temp_val_m1 += eigenvector[basis_inv.at(a_basis)] * m1_dagger.val[j];
               }
            }
            vector_work_m1[i] = temp_val_m1 * fermion_sign_m1;
            
            std::int32_t fermion_sign_m2 = 1;
            if (m_2.tag == blas::CRSTag::FERMION) {
               std::int32_t num_electron = 0;
               for (std::int32_t site = 0; site < one_dim_site_index_2; site++) {
                  num_electron += model_.CalculateNumElectron(ed_utility::CalculateLocalBasis(global_basis, site, dim_onsite));
               }
               if (num_electron%2 == 1) {
                  fermion_sign_m2 = -1;
               }
            }
            
            for (std::int64_t j = m_2.row[local_basis_m2]; j < m_2.row[local_basis_m2 + 1]; ++j) {
               const std::int64_t a_basis = global_basis - (local_basis_m2 - m_2.col[j]) * site_constant_m2;
               if (basis_inv.count(a_basis) != 0) {
                  temp_val_m2 += eigenvector[basis_inv.at(a_basis)]*m_2.val[j];
               }
            }
            vector_work_m2[i] = temp_val_m2*fermion_sign_m2;
         }
         val += blas::CalculateInnerProduct(vector_work_m1, vector_work_m2, num_threads_);
      }
      return val;
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
   
   bool flag_display_info_ = true;
   
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
      diag_params.flag_display_info = flag_display_info_;
      return diag_params;
   }
   
   blas::IIParams<RealType> GetIIParams() const {
      blas::IIParams<RealType> ii_params;
      ii_params.num_threads = num_threads_;
      ii_params.max_step = ii_max_step_;
      ii_params.diag_add = ii_diag_add_;
      ii_params.acc = ii_accuracy_;
      ii_params.flag_display_info = flag_display_info_;
      ii_params.linear_eq_params.num_threads = num_threads_;
      ii_params.linear_eq_params.flag_symmetric_crs = true;
      ii_params.linear_eq_params.flag_use_initial_vec = true;
      ii_params.linear_eq_method = linear_eq_method_;
      ii_params.linear_eq_params.max_step = linear_eq_max_step_;
      ii_params.linear_eq_params.acc = linear_eq_accuracy_;
      ii_params.linear_eq_params.flag_display_info = flag_display_info_;
      return ii_params;
   }
   
   std::unordered_map<std::int64_t, std::int64_t> GenerateInverseBasis(const std::vector<std::int64_t> &basis) const {
      const std::int64_t dim = static_cast<std::int64_t>(basis.size());
      std::unordered_map<std::int64_t, std::int64_t> inverse_basis;
      inverse_basis.reserve(dim + 1);
      for (std::int64_t i = 0; i < dim; ++i) {
         inverse_basis.emplace(basis[i], i);
      }
      return inverse_basis;
   }
   
   blas::DASPM<RealType> GenerateHamiltonian() const {
      const auto start = std::chrono::system_clock::now();
      
      const CQNType target_sector = model_.GetTaretSector();
      const auto &basis = bases_.at(target_sector);
      const auto &basis_inv = inverse_bases_.at(target_sector);
      
      const std::int64_t dim_target = basis.size();
      
      std::vector<std::vector<std::int64_t>> temp_col(dim_target);
      std::vector<std::vector<RealType>> temp_val(dim_target);
      
      blas::DASPM<RealType> ham(dim_target);
            
#pragma omp parallel num_threads(num_threads_)
      {
         ed_utility::ExactDiagMatrixComponents<RealType> components;
         components.site_constant.resize(model_.GetSystemSize());
         for (std::int32_t site = 0; site < model_.GetSystemSize(); ++site) {
            components.site_constant[site] =
            static_cast<std::int64_t>(std::pow(model_.GetDimOnsite(), site));
         }
         components.basis_onsite.resize(model_.GetSystemSize());
         
#pragma omp for schedule(guided)
         for (std::int64_t row = 0; row < dim_target; ++row) {
            ed_utility::GenerateMatrixComponents(&components, basis[row], model_);
            std::vector<std::pair<std::int64_t, RealType>> temp_col_val_list;
            const std::size_t size = components.basis_affected.size();
            for (std::size_t i = 0; i < size; ++i) {
               const std::int64_t a_basis = components.basis_affected[i];
               const RealType val = components.val[i];
               if (basis_inv.count(a_basis) > 0) {
                  const std::int64_t inv = basis_inv.at(a_basis);
                  if ((inv < row && std::abs(val) > std::numeric_limits<RealType>::epsilon()) || inv == row) {
                     temp_col_val_list.push_back({inv, val});
                  }
               }
               else if (basis_inv.count(a_basis) == 0 && std::abs(val) > std::numeric_limits<RealType>::epsilon()) {
                  throw std::runtime_error("Matrix elements are not in the target space");
               }
            }
            std::sort(temp_col_val_list.begin(), temp_col_val_list.end(), [](const auto &a, const auto &b) {
               return a.first < b.first;
            });
            ham.col_val[row] = temp_col_val_list;
            components.val.clear();
            components.basis_affected.clear();
            components.inv_basis_affected.clear();
         }
      }
      
      if (flag_display_info_) {
         const std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
         std::cout << "\rConstruct Hamiltonian: " << elapsed_seconds.count() << " [sec]" << std::endl;
         std::cout << std::flush;
      }
      
      return ham;
   }
   
   std::unordered_set<CQNType, CQNHash> GenerateTargetSector(const CRS &m_1) const {
      std::unordered_set<CQNType, CQNHash> target_q_set;
      for (std::int64_t i = 0; i < m_1.row_dim; ++i) {
         for (std::int64_t j = m_1.row[i]; j < m_1.row[i + 1]; ++j) {
            if (std::abs(m_1.val[j]) > std::numeric_limits<RealType>::epsilon()) {
               target_q_set.emplace(model_.CalculateQNumber(i, m_1.col[j]));
            }
         }
      }
      return target_q_set;
   }
   
   //! @brief Calculate the quantum numbers of excited states that appear when calculating the correlation functions.
   //! @param m_1 The matrix of an onsite operator.
   //! @param m_2 The matrix of an onsite operator.
   //! @return The list of quantum numbers.
   std::vector<CQNType> GenerateTargetSector(const CRS &m_1, const CRS &m_2) const {
      std::unordered_set<CQNType, CQNHash> q_set_m1 = GenerateTargetSector(m_1);
      std::unordered_set<CQNType, CQNHash> q_set_m2 = GenerateTargetSector(m_2);
      
      std::vector<CQNType> target_q_set;
      for (const auto &q_m1 : q_set_m1) {
         if (q_set_m2.count(q_m1) > 0 && model_.ValidateQNumber(q_m1)) {
            target_q_set.push_back(q_m1);
         }
      }
      
      return target_q_set;
   }

   
};


} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_ */
