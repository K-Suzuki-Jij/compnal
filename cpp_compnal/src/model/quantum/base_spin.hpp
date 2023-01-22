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
//  base_spin.hpp
//  compnal
//
//  Created by kohei on 2023/01/22.
//  
//

#ifndef COMPNAL_MODEL_QUANTUM_BASE_SPIN_HPP_
#define COMPNAL_MODEL_QUANTUM_BASE_SPIN_HPP_


#include "../../lattice/all.hpp"
#include "../../blas/all.hpp"
#include "../../utility/hash.hpp"

#include <omp.h>
#include <vector>

namespace compnal {
namespace model {
namespace quantum {

template<class LatticeType, typename RealType>
class BaseSpin {
   static_assert(std::is_floating_point<RealType>::value, "Template parameter RealType must be floating point type");
   
   //! @brief Alias of compressed row strage (CRS) with RealType.
   using CRS = blas::CRS<RealType>;
   
public:
   //! @brief The value type.
   using ValueType = RealType;
   
   //! @brief The type of conserved quantum number (total electron, twice the number of total sz) pair.
   using CQNType = std::int32_t;
   
   using CQNHash = std::hash<std::int32_t>;
      
   BaseSpin(const LatticeType &lattice): lattice_(lattice) {
      SetOnsiteOperator();
   }
   
   //! @brief Set the magnitude of the spin \f$ S \f$.
   //! @param magnitude_spin The magnitude of the spin \f$ S \f$.
   void SetMagnitudeSpin(const double magnitude_spin) {
      if (magnitude_spin <= 0) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Please set magnitude_spin > 0" << std::endl;
         throw std::runtime_error(ss.str());
      }
      if (std::floor(2*magnitude_spin) != 2*magnitude_spin) {
         std::stringstream ss;
         ss << "The input number " << magnitude_spin << " is not half-integer." << std::endl;
         throw std::runtime_error(ss.str());
      }
      magnitude_2spin_ = 2*magnitude_spin;
      dim_onsite_ = magnitude_2spin_ + 1;
      SetOnsiteOperator();
   }
   
   //! @brief Set the total sz.
   //! @param total_sz The total sz \f$ \langle\hat{S}^{z}_{\rm tot}\rangle=\sum^{N}_{i=1}\langle\hat{S}^{z}_{i}\rangle\f$.
   void SetTotalSz(const double total_sz) {
      if (std::floor(2*total_sz) != 2*total_sz) {
         std::stringstream ss;
         ss << "The input number " << total_sz << " is not half-integer." << std::endl;
         throw std::runtime_error(ss.str());
      }
      conserved_quantum_number_ = 2*total_sz;
   }
   
   //! @brief Print the onsite bases.
   void PrintBasisOnsite() const {
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         std::cout << "row " << row << ": |Sz=" << 0.5*magnitude_2spin_ - row << ">" << std::endl;
      }
   }
   
   CQNType GetTaretSector() const {
      return conserved_quantum_number_;
   }
   
   //! @brief Calculate the number of electrons from the input onsite basis.
   //! @param basis_onsite The onsite basis.
   //! @return The number of electrons.
   std::int32_t CalculateNumElectron(const std::int32_t basis_onsite) const {
      if (0 <= basis_onsite && basis_onsite < dim_onsite_) {
         return 0;
      }
      else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid onsite basis" << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Check if there is a subspace specified by the input quantum numbers.
   //! @return ture if there exists corresponding subspace, otherwise false.
   bool ValidateQNumber() const {
      return ValidateQNumber(conserved_quantum_number_);
   }
   
   bool ValidateQNumber(const CQNType conserved_quantum_number) const {
      const auto system_size = lattice_.GetSystemSize();
      const bool c1 = ((system_size*magnitude_2spin_ - conserved_quantum_number) % 2 == 0);
      const bool c2 = (-system_size*magnitude_2spin_ <= conserved_quantum_number);
      const bool c3 = (conserved_quantum_number <= system_size * magnitude_2spin_);
      if (c1 && c2 && c3) {
         return true;
      } else {
         return false;
      }
   }
   
   std::int64_t CalculateTargetDim() const {
      return CalculateTargetDim(conserved_quantum_number_);
   }
   
   std::int64_t CalculateTargetDim(const CQNType conserved_quantum_number) const {
      if (!ValidateQNumber(conserved_quantum_number)) {
         return 0;
      }
      
      const auto system_size = lattice_.GetSystemSize();
      if (system_size <= 0) {
         return 0;
      }
      const std::int32_t max_total_2sz = system_size*magnitude_2spin_;
      std::vector<std::vector<std::int64_t>> dim(system_size, std::vector<std::int64_t>(max_total_2sz + 1));
      for (std::int32_t s = -magnitude_2spin_; s <= magnitude_2spin_; s += 2) {
         dim[0][(s + magnitude_2spin_)/2] = 1;
      }
      for (std::int32_t site = 1; site < system_size; site++) {
         for (std::int32_t s = -magnitude_2spin_; s <= magnitude_2spin_; s += 2) {
            for (std::int32_t s_prev = -magnitude_2spin_*site; s_prev <= magnitude_2spin_*site; s_prev += 2) {
               const std::int64_t a = dim[site][(s + s_prev + magnitude_2spin_*(site + 1))/2];
               const std::int64_t b = dim[site - 1][(s_prev + magnitude_2spin_*site)/2];
               if (a >= INT64_MAX - b) {
                  throw std::overflow_error("Overflow detected for sumation using uint64_t");
               }
               dim[site][(s + s_prev + magnitude_2spin_*(site + 1))/2] = a + b;
            }
         }
      }
      return dim[system_size - 1][(conserved_quantum_number + max_total_2sz)/2];
   }
   
   std::vector<std::int64_t> GenerateBasis(const std::int32_t num_threads = 1) const {
      return GenerateBasis(conserved_quantum_number_, num_threads);
   }
   
   std::vector<std::int64_t> GenerateBasis(const CQNType conserved_quantum_number,
                                           const std::int32_t num_threads = 1) const {
      
      if (!ValidateQNumber(conserved_quantum_number)) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid parameters: system_size(" << lattice_.GetSystemSize();
         ss << ") or magnitude_spin(" << 0.5*magnitude_2spin_ << ") or total_sz(" << 0.5*conserved_quantum_number << ")" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const auto system_size = lattice_.GetSystemSize();
      const double magnitude_spin = 0.5*magnitude_2spin_;
      const double total_sz = 0.5*conserved_quantum_number;
      auto partition_integers = utility::GenerateIntegerPartition(static_cast<std::int32_t>(system_size*magnitude_spin - total_sz),
                                                                  magnitude_2spin_, system_size);
      
      for (auto &&it : partition_integers) {
         it.resize(system_size, 0);
      }
      
      std::vector<std::int64_t> site_constant(system_size);
      for (std::int32_t site = 0; site < system_size; ++site) {
         site_constant[site] = static_cast<std::int64_t>(std::pow(dim_onsite_, site));
      }
      
      const std::int64_t dim_target = CalculateTargetDim(conserved_quantum_number);
      std::vector<std::int64_t> basis;
      basis.reserve(dim_target);
      
      std::vector<std::vector<std::int64_t>> temp_basis(num_threads);
      for (const auto &integer_list : partition_integers) {
         const std::int64_t size = utility::CalculateNumPermutation(integer_list);
#pragma omp parallel num_threads(num_threads)
         {
            const std::int32_t thread_num = omp_get_thread_num();
            const std::int64_t loop_begin = thread_num * size / num_threads;
            const std::int64_t loop_end = (thread_num + 1) * size / num_threads;
            std::vector<std::int32_t> n_th_integer_list = utility::GenerateNthPermutation(integer_list, loop_begin);
            for (std::int64_t j = loop_begin; j < loop_end; ++j) {
               std::int64_t basis_global = 0;
               for (std::size_t k = 0; k < n_th_integer_list.size(); ++k) {
                  basis_global += n_th_integer_list[k] * site_constant[k];
               }
               temp_basis[thread_num].push_back(basis_global);
               std::next_permutation(n_th_integer_list.begin(), n_th_integer_list.end());
            }
         }
      }
      for (auto &&it : temp_basis) {
         basis.insert(basis.end(), it.begin(), it.end());
         std::vector<std::int64_t>().swap(it);
      }
      
      basis.shrink_to_fit();
      
      if (static_cast<std::int64_t>(basis.size()) != dim_target) {
         std::stringstream ss;
         ss << "Unknown error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "basis.size()=" << basis.size() << ", but dim_target=" << dim_target << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      std::sort(basis.begin(), basis.end());
      
      return basis;
   }
   
   //! @brief Calculate sectors generated by an onsite operator.
   //! @tparam IntegerType Value type int row and col.
   //! @param row The row in the matrix representation of an onsite operator.
   //! @param col The column in the matrix representation of an onsite operator.
   //! @return Sectors generated by an onsite operator.
   template <typename IntegerType>
   CQNType CalculateQNumber(const IntegerType row, const IntegerType col) const {
      static_assert(std::is_integral<IntegerType>::value, "Template parameter IntegerType must be integer type");
      if (row < 0 || col < 0 || dim_onsite_ <= row || dim_onsite_ <= col) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid parameters" << std::endl;
         throw std::runtime_error(ss.str());
      }
      return static_cast<CQNType>(2*(col - row) + conserved_quantum_number_);
   }
   
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
   
   std::int32_t GetDimOnsite() const { return dim_onsite_; }
   
   double GetTotalSz() const { return 0.5*conserved_quantum_number_; }
   
   //! @brief Get the magnitude of the spin \f$ S\f$.
   //! @return The magnitude of the spin \f$ S\f$.
   double GetMagnitudeSpin() const { return 0.5*magnitude_2spin_; }
   
   //! @brief Get the spin-\f$ S\f$ operator for the x-direction \f$ \hat{s}^{x}\f$.
   //! @return The matrix of \f$ \hat{s}^{x}\f$.
   const CRS &GetOnsiteOperatorSx() const { return onsite_operator_sx_; }
   
   //! @brief Get the spin-\f$ S\f$ operator for the y-direction \f$ i\hat{s}^{y}\f$ with \f$ i\f$ being the imaginary
   //! unit.
   //! @return The matrix of \f$ i\hat{s}^{y}\f$.
   const CRS &GetOnsiteOperatoriSy() const { return onsite_operator_isy_; }
   
   //! @brief Get the spin-\f$ S\f$ operator for the z-direction \f$ \hat{s}^{z}\f$.
   //! @return The matrix of \f$ \hat{s}^{z}\f$.
   const CRS &GetOnsiteOperatorSz() const { return onsite_operator_sz_; }
   
   //! @brief Get the spin-\f$ S\f$ raising operator \f$ \hat{s}^{+}\f$.
   //! @return The matrix of \f$ \hat{s}^{+}\f$.
   const CRS &GetOnsiteOperatorSp() const { return onsite_operator_sp_; }
   
   //! @brief Get the spin-\f$ S\f$ lowering operator \f$ \hat{s}^{-}\f$.
   //! @return The matrix of \f$ \hat{s}^{-}\f$.
   const CRS &GetOnsiteOperatorSm() const { return onsite_operator_sm_; }
   
private:
   //! @brief The linear interaction.
   const LatticeType lattice_;
   
   CQNType conserved_quantum_number_ = 0;
   
   //! @brief The magnitude of the spin \f$ S\f$.
   std::int32_t magnitude_2spin_ = 1;
   
   //! @brief The dimension of the local Hilbert space, \f$ 2S + 1\f$.
   std::int32_t dim_onsite_ = magnitude_2spin_ + 1;
   
   //! @brief The spin-\f$ S\f$ operator for the x-direction \f$ \hat{s}^{x}\f$.
   CRS onsite_operator_sx_;
   
   //! @brief The spin-\f$ S\f$ operator for the y-direction \f$ i\hat{s}^{y}\f$ with \f$ i\f$ being the imaginary unit.
   CRS onsite_operator_isy_;
   
   //! @brief The spin-\f$ S\f$ operator for the z-direction \f$ \hat{s}^{z}\f$.
   CRS onsite_operator_sz_;
   
   //! @brief The spin-\f$ S\f$ raising operator \f$ \hat{s}^{+}\f$.
   CRS onsite_operator_sp_;
   
   //! @brief The the spin-\f$ S\f$ raising operator \f$ \hat{s}^{-}\f$.
   CRS onsite_operator_sm_;
   
   //! @brief Set onsite operators.
   void SetOnsiteOperator() {
      onsite_operator_sx_ = CreateOnsiteOperatorSx();
      onsite_operator_isy_ = CreateOnsiteOperatoriSy();
      onsite_operator_sz_ = CreateOnsiteOperatorSz();
      onsite_operator_sp_ = CreateOnsiteOperatorSp();
      onsite_operator_sm_ = CreateOnsiteOperatorSm();
   }
   
   CRS CreateOnsiteOperatorSz() {
      CRS matrix(dim_onsite_, dim_onsite_);
      
      for (std::int32_t row = 0; row < dim_onsite_; ++row) {
         const RealType val = 0.5*magnitude_2spin_ - row;
         if (val != 0.0) {
            matrix.val.push_back(val);
            matrix.col.push_back(row);
         }
         matrix.row[row + 1] = matrix.col.size();
      }
      matrix.tag = blas::CRSTag::BOSON;
      return matrix;
   }
   
   CRS CreateOnsiteOperatorSp() {
      const RealType constant = 2.0;
      CRS matrix(dim_onsite_, dim_onsite_);
      for (std::int32_t row = 1; row < dim_onsite_; ++row) {
         matrix.val.push_back(std::sqrt((0.5*magnitude_2spin_ + 1)*constant*row - row*(row + 1)));
         matrix.col.push_back(row);
         matrix.row[row] = matrix.col.size();
      }
      matrix.row[dim_onsite_] = matrix.col.size();
      matrix.tag = blas::CRSTag::BOSON;
      return matrix;
   }
   
   //! @brief Generate the spin-\f$ S\f$ raising operator \f$ \hat{s}^{-}\f$.
   //! @return The matrix of \f$ \hat{s}^{-}\f$.
   CRS CreateOnsiteOperatorSm() {
      CRS matrix(dim_onsite_, dim_onsite_);
      const RealType constant = 2.0;
      for (std::int32_t row = 1; row < dim_onsite_; ++row) {
         matrix.val.push_back(std::sqrt((0.5*magnitude_2spin_ + 1)*constant*row - row*(row + 1)));
         matrix.col.push_back(row - 1);
         matrix.row[row + 1] = matrix.col.size();
      }
      matrix.tag = blas::CRSTag::BOSON;
      return matrix;
   }
   
   CRS CreateOnsiteOperatorSx() {
      return RealType{0.5}*(CreateOnsiteOperatorSp() + CreateOnsiteOperatorSm());
   }
   
   CRS CreateOnsiteOperatoriSy() {
      return RealType{0.5}*(CreateOnsiteOperatorSp() - CreateOnsiteOperatorSm());
   }
   
   
};


} // namespace quantum
} // namespace model
} // namespace compnal

#endif /* COMPNAL_MODEL_QUANTUM_BASE_SPIN_HPP_ */
