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

template<class ModelType>
class ExactDiag {
   using RealType = typename ModelType::ValueType;
   using CQNType = typename ModelType::CQNType;
   using CQNHash = typename ModelType::CQNHash;
   using CRS = blas::CRS<RealType>;
   
public:

   ExactDiag(const ModelType &model_): model_(model_) {}
   
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
      
      if (diag_method == blas::DiagAlgorithm::LANCZOS) {
         blas::EigendecompositionLanczos(&eigenvalues_[0], &eigenvectors_[0], ham, diag_params, {});
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
      
      blas::InverseIteration(&ham, &eigenvectors_[0], eigenvalues_[0], {}, ii_params);
      
   }
   
   
   
   
private:
   const ModelType model_;
   std::vector<RealType> eigenvalues_;
   std::vector<std::vector<RealType>> eigenvectors_;
   
   std::unordered_map<CQNType, std::vector<std::int64_t>, CQNHash> bases_;
   std::unordered_map<CQNType, std::unordered_map<std::int64_t, std::int64_t>, CQNHash> inverse_bases_;
   
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
   
   CRS GenerateHamiltonian() const {
      
      const CQNType target_sector = model_.GetTaretSector();
      const auto &basis = bases_.at(target_sector);
      const auto &basis_inv = inverse_bases_.at(target_sector);
      
      const std::int64_t dim_target = basis.size();
      std::int64_t num_total_elements = 0;
      
      const int num_threads = omp_get_max_threads();
      std::vector<ExactDiagMatrixComponents<RealType>> components(num_threads);
      
      for (int thread_num = 0; thread_num < num_threads; ++thread_num) {
         components[thread_num].site_constant.resize(model_.GetSystemSize());
         for (int site = 0; site < model_.GetSystemSize(); ++site) {
            components[thread_num].site_constant[site] =
            static_cast<std::int64_t>(std::pow(model_.GetDimOnsite(), site));
         }
         components[thread_num].basis_onsite.resize(model_.GetSystemSize());
      }
      
      std::vector<std::int64_t> num_row_element(dim_target + 1);
      
#pragma omp parallel for
      for (std::int64_t row = 0; row < dim_target; ++row) {
         const int thread_num = omp_get_thread_num();
         GenerateMatrixComponents(&components[thread_num], basis[row], model_);
         const std::size_t size = components[thread_num].basis_affected.size();
         for (std::size_t i = 0; i < size; ++i) {
            const std::int64_t a_basis = components[thread_num].basis_affected[i];
            const RealType val = components[thread_num].val[i];
            if (basis_inv.count(a_basis) > 0) {
               const std::int64_t inv = basis_inv.at(a_basis);
               if ((inv < row && std::abs(val) > components[thread_num].zero_precision) || inv == row) {
                  num_row_element[row + 1]++;
               }
            } else if (basis_inv.count(a_basis) == 0 && std::abs(val) > components[thread_num].zero_precision) {
               throw std::runtime_error("Matrix elements are not in the target space");
            }
         }
         components[thread_num].val.clear();
         components[thread_num].basis_affected.clear();
         components[thread_num].inv_basis_affected.clear();
      }
      
#pragma omp parallel for reduction(+: num_total_elements)
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
      
#pragma omp parallel for
      for (std::int64_t row = 0; row < dim_target; ++row) {
         const int thread_num = omp_get_thread_num();
         GenerateMatrixComponents(&components[thread_num], basis[row], model_);
         const std::size_t size = components[thread_num].basis_affected.size();
         for (std::size_t i = 0; i < size; ++i) {
            const std::int64_t a_basis = components[thread_num].basis_affected[i];
            const RealType val = components[thread_num].val[i];
            if (basis_inv.count(a_basis) > 0) {
               const std::int64_t inv = basis_inv.at(a_basis);
               if ((inv < row && std::abs(val) > components[thread_num].zero_precision) || inv == row) {
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
         
      return ham;
   }
   
   
   
   
};


} // namespace solver
} // namespace compnal

#endif /* COMPNAL_SOLVER_EXACT_DIAGONALIZATION_HPP_ */
