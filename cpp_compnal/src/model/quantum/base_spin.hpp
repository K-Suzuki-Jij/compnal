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
      return ValidateQNumber(lattice_.GetSystemSize(), magnitude_2spin_, conserved_quantum_number);
   }
   
   static bool ValidateQNumber(const std::int32_t system_size,
                               const std::int32_t magnitude_2spin,
                               const std::int32_t total_2sz) {
      const bool c1 = ((system_size*magnitude_2spin - total_2sz)%2 == 0);
      const bool c2 = (-system_size*magnitude_2spin <= total_2sz);
      const bool c3 = (total_2sz <= system_size * magnitude_2spin);
      if (c1 && c2 && c3) {
         return true;
      }
      else {
         return false;
      }
   }
   
   std::int64_t CalculateTargetDim() const {
      return CalculateTargetDim(conserved_quantum_number_);
   }
   
   std::int64_t CalculateTargetDim(const CQNType conserved_quantum_number) const {
      return CalculateTargetDim(lattice_.GetSystemSize(), magnitude_2spin_, conserved_quantum_number);
   }
   
   static std::int64_t CalculateTargetDim(const std::int32_t system_size,
                                          const std::int32_t magnitude_2spin,
                                          const std::int32_t total_2sz) {
      if (!ValidateQNumber(system_size, magnitude_2spin, total_2sz)) {
         return 0;
      }
      
      if (system_size <= 0) {
         return 0;
      }
      
      const std::int32_t max_total_2sz = system_size*magnitude_2spin;
      std::vector<std::vector<std::int64_t>> dim(system_size, std::vector<std::int64_t>(max_total_2sz + 1));
      for (std::int32_t s = -magnitude_2spin; s <= magnitude_2spin; s += 2) {
         dim[0][(s + magnitude_2spin)/2] = 1;
      }
      for (std::int32_t site = 1; site < system_size; site++) {
         for (std::int32_t s = -magnitude_2spin; s <= magnitude_2spin; s += 2) {
            for (std::int32_t s_prev = -magnitude_2spin*site; s_prev <= magnitude_2spin*site; s_prev += 2) {
               const std::int64_t a = dim[site][(s + s_prev + magnitude_2spin*(site + 1))/2];
               const std::int64_t b = dim[site - 1][(s_prev + magnitude_2spin*site)/2];
               if (a >= INT64_MAX - b) {
                  throw std::overflow_error("Overflow detected for sumation using uint64_t");
               }
               dim[site][(s + s_prev + magnitude_2spin*(site + 1))/2] = a + b;
            }
         }
      }
      return dim[system_size - 1][(total_2sz + max_total_2sz)/2];
   }
   
   
   std::vector<std::int64_t> GenerateBasis() const {
      return GenerateBasis(conserved_quantum_number_);
   }
   
   std::vector<std::int64_t> GenerateBasis(const CQNType conserved_quantum_number) const {
      std::vector<std::int64_t> site_constant(lattice_.GetSystemSize());
      for (std::int32_t site = 0; site < lattice_.GetSystemSize(); ++site) {
         site_constant[site] = static_cast<std::int64_t>(std::pow(dim_onsite_, site));
      }
      return GenerateBasis(conserved_quantum_number, lattice_.GetSystemSize(), magnitude_2spin_, site_constant);
   }
   
   
   static std::vector<std::int64_t> GenerateBasis(const CQNType conserved_quantum_number,
                                                  const std::int32_t system_size,
                                                  const std::int32_t magnitude_2spin,
                                                  const std::vector<std::int64_t> site_constant) {
      
      if (!ValidateQNumber(system_size, magnitude_2spin, conserved_quantum_number)) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Invalid parameters: system_size(" << system_size;
         ss << ") or magnitude_spin(" << 0.5*magnitude_2spin << ") or total_sz(" << 0.5*conserved_quantum_number << ")" << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      if (site_constant.size() != system_size) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      const std::int32_t partitioned_number = system_size*magnitude_2spin/2 - conserved_quantum_number/2;
      std::vector<std::vector<std::int32_t>> partition_integers;
      std::vector<std::int32_t> initi_list;
      utility::GenerateIntegerPartition(&partition_integers, initi_list, partitioned_number, magnitude_2spin, system_size);
      
      for (auto &&it : partition_integers) {
         it.resize(system_size, 0);
      }
      
      const std::int64_t dim_target = CalculateTargetDim(system_size, magnitude_2spin, conserved_quantum_number);
      std::vector<std::int64_t> basis;
      basis.reserve(dim_target);
      
      for (std::size_t i = 0; i < partition_integers.size(); ++i) {
         auto &integer_list = partition_integers[i];
         std::sort(integer_list.begin(), integer_list.end());
         do {
            std::int64_t basis_global = 0;
            for (std::size_t j = 0; j < integer_list.size(); ++j) {
               basis_global += integer_list[j]*site_constant[j];
            }
            basis.push_back(basis_global);
         } while (std::next_permutation(integer_list.begin(), integer_list.end()));
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
   template<typename IntegerType>
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
