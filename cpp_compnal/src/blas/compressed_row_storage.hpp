//
//  Copyright 2022 Kohei Suzuki
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
//  compressed_row_storage.hpp
//  compnal
//
//  Created by kohei on 2022/10/06.
//  
//

#ifndef COMPNAL_BLAS_COMPRESSED_ROW_STORAGE_HPP_
#define COMPNAL_BLAS_COMPRESSED_ROW_STORAGE_HPP_

#include "../utility/sort.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace compnal {
namespace blas {

//! @brief Enumerated type to represent Fermion, Boson, etc..
enum CRSTag {
   
   //! @brief No taged.
   NONE = 0,
   
   //! @brief Fermionic operator.
   FERMION = 1,
   
   //! @brief Bosonic operator.
   BOSON = 2,
   
   //! @brief Sum of Fermion and Boson.
   MIX = 3
};

//! @brief Operator overloading: output operator.
//! @param os Ostream object.
//! @param tag CRSTag
std::ostream &operator<<(std::ostream &os, const CRSTag &tag) {
   if (tag == CRSTag::NONE) {
      os << "CRSTag::NONE";
   } else if (tag == CRSTag::FERMION) {
      os << "CRSTag::FERMION";
   } else if (tag == CRSTag::BOSON) {
      os << "CRSTag::BOSON";
   } else if (tag == CRSTag::MIX) {
      os << "CRSTag::MIX";
   } else {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "Unknown CRSTag detected.";
      throw std::runtime_error(ss.str());
   }
   return os;
}


template<typename ElementType>
struct CRS {
   
   //! @brief Alias of ElementType.
   using ValueType = ElementType;
   
   //------------------------------------------------------------------
   //----------------------Public Member Variables---------------------
   //------------------------------------------------------------------
   //! @brief The dimension of the row.
   std::int64_t row_dim = 0;
   
   //! @brief The dimension of the colmun.
   std::int64_t col_dim = 0;
   
   //! @brief The locations in the val vector that start a row.
   std::vector<std::int64_t> row = {0};
   
   //! @brief The column indexes of the elements in the val vector.
   std::vector<std::int64_t> col;
   
   //! @brief The values of the nonzero elements of the matrix.
   std::vector<ElementType> val;
   
   //! @brief Type of the the matrix.
   CRSTag tag = CRSTag::NONE;
   
   //! @brief Matrix name.
   std::string name = "";
   
   //------------------------------------------------------------------
   //---------------------------Constructors---------------------------
   //------------------------------------------------------------------
   //! @brief Constructor of CRS class.
   CRS() {}
   
   //! @brief Constructor of CRS class.
   //! @param row_dim_in The dimension of the row.
   //! @param col_dim_in The dimension of the colmun.
   //! @param tag_in Type of the the matrix. Defaults to CRSTag::NONE.
   CRS(const std::int64_t row_dim_in, const std::int64_t col_dim_in, const CRSTag tag_in = CRSTag::NONE):
   row_dim(row_dim_in), col_dim(col_dim_in), tag(tag_in) {
      row.resize(row_dim_in + 1);
   }
   
   //! @brief Constructor of CRS class.
   //! @tparam T The value type of the vector elements.
   //! @param mat_vec The matrix of two dimensional vectors.
   //! @param tag_in Type of the the matrix. Defaults to CRSTag::NONE.
   template<typename T = ElementType>
   explicit CRS(const std::vector<std::vector<T>> &mat_vec, const CRSTag tag_in = CRSTag::NONE) {
      this->row_dim = mat_vec.size();
      this->col_dim = 0;
      this->row.resize(this->row_dim + 1);
      this->row[0] = 0;
      for (std::int64_t i = 0; i < this->row_dim; ++i) {
         const std::int64_t size = static_cast<std::int64_t>(mat_vec[i].size());
         for (std::int64_t j = 0; j < size; ++j) {
            if (mat_vec[i][j] != 0.0) {
               this->col.push_back(j);
               this->val.push_back(mat_vec[i][j]);
            }
            if (this->col_dim < j + 1) {
               this->col_dim = j + 1;
            }
         }
         this->row[i + 1] = this->col.size();
      }
      this->tag = tag_in;
   }
   
   //! @brief Copy constructor of CRS class.
   //! @tparam T Value type of the matrix elements.
   //! @param matrix The matrix.
   template <typename T>
   CRS(const CRS<T> &matrix) {
      Assign(matrix);
   }
   
   //! @brief Assign CRS object.
   //! @tparam T Value type of the matrix elements.
   //! @param matrix The matrix.
   template <typename T>
   void Assign(const CRS<T> &matrix, const std::int32_t num_threads = utility::DEFAULT_NUM_THREADS) {
      this->row_dim = matrix.row_dim;
      this->col_dim = matrix.col_dim;
      this->row.resize(matrix.row_dim + 1);
      
      this->col.resize(matrix.col.size());
      this->val.resize(matrix.col.size());
      
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::int64_t i = 0; i <= matrix.row_dim; ++i) {
         this->row[i] = matrix.row[i];
      }
      
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::size_t i = 0; i < matrix.col.size(); ++i) {
         this->col[i] = matrix.col[i];
         this->val[i] = matrix.val[i];
      }
      this->tag = matrix.tag;
      this->name = matrix.name;
   }
   
   template<typename T>
   void MultiplyByScalar(const T coeff, const std::int32_t num_threads = utility::DEFAULT_NUM_THREADS) {
      if (std::abs(coeff) < std::numeric_limits<ElementType>::epsilon() < T{0}) {
#pragma omp parallel for schedule(guided) num_threads(num_threads)
         for (std::int64_t i = 0; i < this->row_dim; ++i) {
            this->row[i + 1] = 0;
         }
         this->col.clear();
         this->val.clear();
      }
      else {
#pragma omp parallel for schedule(guided) num_threads(num_threads)
         for (std::size_t i = 0; i < this->col.size(); ++i) {
            this->val[i] *= coeff;
         }
      }
   }
   
   void SortCol(const std::int32_t num_threads = utility::DEFAULT_NUM_THREADS) {
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (std::int64_t i = 0; i < this->row_dim; i++) {
         utility::QuickSortVector(&this->col, &this->val, this->row[i], this->row[i + 1]);
      }
   }
   
   //! @brief Clear the elements and free memory.
    void Free() {
       this->row_dim = 0;
       this->col_dim = 0;
       std::vector<std::int64_t>().swap(this->row);
       std::vector<std::int64_t>().swap(this->col);
       std::vector<ElementType>().swap(this->val);
       this->row.push_back(0);
       this->tag = CRSTag::NONE;
       this->name = "";
    }
   
   //! @brief Check if the matrix is symmetric or not within the threshold.
   //! @param threshold The threshold. Defaults to 10^-15.
   //! @param flag_display_info Display information if the matrix is not symmetric.
   //! @return Return true if the matrix is symmetric, otherwise false.
   bool CheckSymmetric(const ElementType threshold = std::numeric_limits<ElementType>::epsilon(),
                       const bool flag_display_info = true) const {
      if (this->row_dim != this->col_dim) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "The matrix is not square one." << std::endl;
         throw std::runtime_error(ss.str());
      }
      // Check sorted
      bool flag_sorted = true;
      for (std::int64_t i = 0; i < this->row_dim; ++i) {
         for (std::int64_t j = this->row[i]; j < this->row[i + 1] - 1; ++j) {
            if (this->col[j] >= this->col[j + 1]) {
               flag_sorted = false;
               break;
            }
         }
      }
      if (!flag_sorted) {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "The colmun indexes of the matrix is not sorted." << std::endl;
         throw std::runtime_error(ss.str());
      }
      for (std::int64_t i = 0; i < this->row_dim; ++i) {
         for (std::int64_t j = this->row[i]; j < this->row[i + 1]; ++j) {
            const auto iter_begin = this->col.begin() + this->row[this->col[j]];
            const auto iter_end = this->col.begin() + this->row[this->col[j] + 1];
            const auto iter_find = std::lower_bound(iter_begin, iter_end, i);
            if (iter_find == iter_end || *iter_find != i) {
               if (flag_display_info) {
                  std::cout << "The input matrix is not symmetric." << std::endl;
                  std::cout << "Corresponding element does not exist." << std::endl;
                  std::cout << "row=" << i << ", col=" << col[j] << ", val=" << val[j] << std::endl;
               }
               return false;
            }
            const auto inv = std::distance(this->col.begin(), iter_find);
            if (std::abs(this->val[j] - this->val[inv]) > threshold) {
               if (flag_display_info) {
                  std::cout << "The input matrix is not symmetric." << std::endl;
                  std::cout << "M[" << i << "][" << this->col[j] << "]=" << this->val[j] << ", " << this->val[inv]
                  << "=M[" << this->col[j] << "][" << i << "]" << std::endl;
               }
               return false;
            }
         }
      }
      return true;
   }
   
   //! @brief Print the matrix.
   void Print(const std::string display_name = "") const {
      if (this->name != "") {
         std::cout << "Name: " << this->name << std::endl;
      }
      std::cout << this->tag << std::endl;
      std::cout << std::fixed;
      std::cout << std::setprecision(std::numeric_limits<ElementType>::max_digits10);
      for (std::int64_t i = 0; i < this->row_dim; ++i) {
         for (std::int64_t j = this->row.at(i); j < this->row.at(i + 1); ++j) {
            std::cout << display_name << "[";
            std::cout << std::noshowpos << std::left << std::setw(3) << i << "][";
            std::cout << std::left << std::setw(3) << this->col[j] << "]=";
            std::cout << std::showpos << this->val[j] << std::endl;
         }
      }
      std::cout << std::noshowpos;
   }
   
   
   //! @brief Print the information about the matrix.
   void PrintInfo(const std::string display_name = "Matrix") const {
      std::cout << std::fixed;
      std::cout << std::setprecision(std::numeric_limits<ElementType>::max_digits10);
      std::cout << "Print information about CRS: " << display_name << std::endl;
      std::cout << this->tag << std::endl;
      std::cout << "row_dim = " << this->row_dim << std::endl;
      std::cout << "col_dim = " << this->col_dim << std::endl;
      for (std::size_t i = 0; i < this->row.size(); ++i) {
         std::cout << "row[" << i << "] = " << this->row.at(i) << std::endl;
      }
      for (std::size_t i = 0; i < this->col.size(); ++i) {
         std::cout << "col[" << i << "] = " << this->col.at(i) << std::endl;
      }
      for (std::size_t i = 0; i < this->val.size(); ++i) {
         std::cout << "val[" << i << "] = " << this->val.at(i) << std::endl;
      }
   }
   
   //------------------------------------------------------------------
   //-----------------------Operator Overloading-----------------------
   //------------------------------------------------------------------
   //! @brief Operator overloading: unary plus operator.
   //! @return CRS object, \f$ +\hat{M} \f$.
   CRS operator+() const { return *this; }
   
   //! @brief Operator overloading: unary negation operator.
   //! @return CRS object, \f$ -\hat{M} \f$.
   CRS operator-() const { return CalculateScalarMatrixProduct(-1, *this); }
   
   //! @brief Operator overloading: compound assignment plus operator.
   //! @tparam T Value type of the right-hand side.
   //! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs}\f$.
   //! @return CRS object, \f$ \hat{M} + \hat{M}_{\rm rhs} \f$.
   template <typename T>
   CRS &operator+=(const CRS<T> rhs) {
      return *this = *this + rhs;
   }
   
   //! @brief Operator overloading: compound assignment subtraction operator.
   //! @tparam T Value type of the right-hand side.
   //! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm
   //! rhs}\f$.
   //! @return CRS object, \f$ \hat{M} - \hat{M}_{\rm rhs} \f$.
   template <typename T>
   CRS &operator-=(const CRS<T> rhs) {
      return *this = *this - rhs;
   }
   
   //! @brief Operator overloading: compound assignment multiplication operator.
   //! @tparam T Value type of the right-hand side.
   //! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm
   //! rhs}\f$.
   //! @return CRS object, \f$ \hat{M}\hat{M}_{\rm rhs} \f$.
   template <typename T>
   CRS &operator*=(const CRS<T> rhs) {
      return *this = *this * rhs;
   }
   
   //! @brief Operator overloading: assignment operator.
   //! @tparam T Value type of the matrix elements.
   //! @param matrix The CRS object to be assigned.
   //! @return The CRS object.
   template <typename T>
   CRS &operator=(const CRS<T> &matrix) & {
      Assign(matrix);
      return *this;
   }
   
};


//------------------------------------------------------------------
//----------------------Operator overloading------------------------
//------------------------------------------------------------------
//! @brief Operator overloading: addition operator.
//! @tparam T1 Value type of the left-hand side CRS object.
//! @tparam T2 Value type of the right-hand side CRS object.
//! @param lhs The CRS object of the left-hand side, \f$ \hat{M}_{\rm lhs} \f$.
//! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs} \f$.
//! @return CRS object, \f$ \hat{M}_{\rm lhs} + \hat{M}_{\rm rhs}\f$
template <typename T1, typename T2>
auto operator+(const CRS<T1> &lhs, const CRS<T2> &rhs) -> CRS<decltype(std::declval<T1>() + std::declval<T2>())> {
   return CalculateMatrixMatrixSum(T1{1}, lhs, T2{1}, rhs);
}

//! @brief Operator overloading: subtraction operator.
//! @tparam T1 Value type of the left-hand side CRS object.
//! @tparam T2 Value type of the right-hand side CRS object.
//! @param lhs The CRS object of the left-hand side, \f$ \hat{M}_{\rm lhs} \f$.
//! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs} \f$.
//! @return CRS object, \f$ \hat{M}_{\rm lhs} - \hat{M}_{\rm rhs}\f$
template <typename T1, typename T2>
auto operator-(const CRS<T1> &lhs, const CRS<T2> &rhs) -> CRS<decltype(std::declval<T1>() - std::declval<T2>())> {
   return CalculateMatrixMatrixSum(T1{1}, lhs, T2{-1}, rhs);
}

//! @brief Operator overloading: multiplication operator.
//! @tparam T1 Value type of the left-hand side CRS object.
//! @tparam T2 Value type of the right-hand side CRS object.
//! @param lhs The CRS object of the left-hand side, \f$ \hat{M}_{\rm lhs} \f$.
//! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs} \f$.
//! @return CRS object, \f$ \hat{M}_{\rm lhs}\hat{M}_{\rm rhs}\f$
template <typename T1, typename T2>
auto operator*(const CRS<T1> &lhs, const CRS<T2> &rhs) -> CRS<decltype(std::declval<T1>() * std::declval<T2>())> {
   return CalculateMatrixMatrixProduct(T1{1}, lhs, rhs);
}

//! @brief Operator overloading: multiplication operator.
//! @tparam T1 Value type of the left-hand side.
//! @tparam T2 Value type of the right-hand side CRS object.
//! @param lhs The value of the left-hand side, \f$ c_{\rm lhs}\f$
//! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs} \f$.
//! @return CRS object, \f$ c_{\rm lhs}\hat{M}_{\rm rhs}\f$
template <typename T1, typename T2>
auto operator*(const T1 lhs, const CRS<T2> &rhs) -> CRS<decltype(std::declval<T1>() * std::declval<T2>())> {
   return CalculateScalarMatrixProduct(lhs, rhs);
}

//! @brief Operator overloading: multiplication operator.
//! @tparam T1 Value type of the left-hand side CRS object.
//! @tparam T2 Value type of the right-hand side.
//! @param lhs The CRS object of the left-hand side, \f$ \hat{M}_{\rm lhs} \f$.
//! @param rhs The value of the right-hand side, \f$ c_{\rm rhs}\f$
//! @return CRS object, \f$ \hat{M}_{\rm lhs}c_{\rm rhs}=c_{\rm rhs}\hat{M}_{\rm
//! lhs}\f$
template <typename T1, typename T2>
auto operator*(const CRS<T1> &lhs, const T2 rhs) -> CRS<decltype(std::declval<T1>() * std::declval<T2>())> {
   return CalculateScalarMatrixProduct(rhs, lhs);
}

//! @brief Operator overloading: equality operator.
//! @tparam T1 Value type of the left-hand side CRS object.
//! @tparam T2 Value type of the right-hand side CRS object.
//! @param lhs The CRS object of the left-hand side, \f$ \hat{M}_{\rm lhs} \f$.
//! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs} \f$.
//! @return Return true if \f$ hat{M}_{\rm lhs}=\hat{M}_{\rm rhs}\f$, otherwise
//! false.
template <typename T1, typename T2>
bool operator==(const CRS<T1> &lhs, const CRS<T2> &rhs) {
   if (lhs.row_dim != rhs.row_dim) {
      return false;
   }
   if (lhs.col_dim != rhs.col_dim) {
      return false;
   }
   if (lhs.row.size() != rhs.row.size()) {
      return false;
   }
   if (lhs.tag != rhs.tag) {
      return false;
   }
   if (lhs.col.size() != rhs.col.size()) {
      return false;
   }
   if (lhs.val.size() != rhs.val.size()) {
      return false;
   }
   if (lhs.name != rhs.name) {
      return false;
   }
   for (std::size_t i = 0; i < lhs.row.size(); ++i) {
      if (lhs.row[i] != rhs.row[i]) {
         return false;
      }
   }
   for (std::size_t i = 0; i < lhs.col.size(); ++i) {
      if (lhs.col[i] != rhs.col[i]) {
         return false;
      }
   }
   for (std::size_t i = 0; i < lhs.val.size(); ++i) {
      if (lhs.val[i] != rhs.val[i]) {
         return false;
      }
   }
   return true;
}

//! @brief Operator overloading: equality operator.
//! @tparam T1 Value type of the left-hand side CRS object.
//! @tparam T2 Value type of the right-hand side CRS object.
//! @param lhs The CRS object of the left-hand side, \f$ \hat{M}_{\rm lhs} \f$.
//! @param rhs The CRS object of the right-hand side, \f$ \hat{M}_{\rm rhs} \f$.
//! @return Return true if \f$ \hat{M}_{\rm lhs}\neq\hat{M}_{\rm rhs}\f$,
//! otherwise false.
template <typename T1, typename T2>
bool operator!=(const CRS<T1> &lhs, const CRS<T2> &rhs) {
   return !(lhs == rhs);
}

//------------------------------------------------------------------
//----------------Operator overloading: I/O Stream------------------
//------------------------------------------------------------------
//! @brief Operator overloading: output operator.
//! @tparam T Value type of CRS matrix.
//! @param os Ostream object.
//! @param m The matrix as CRS form.
template <typename T>
std::ostream &operator<<(std::ostream &os, const CRS<T> &m) {
   os << m.tag << std::endl;
   os << std::fixed;
   os << std::setprecision(std::numeric_limits<T>::max_digits10);
   for (std::int64_t i = 0; i < m.row_dim; ++i) {
      for (std::int64_t j = m.row.at(i); j < m.row.at(i + 1); ++j) {
         os << m.name;
         os << "[";
         os << std::noshowpos << std::left << std::setw(3) << i << "][";
         os << std::left << std::setw(3) << m.col[j] << "]=";
         os << std::showpos << m.val[j] << std::endl;
      }
   }
   return os;
}

//------------------------------------------------------------------
//------------------------Global Functions--------------------------
//------------------------------------------------------------------
//! @brief Generate transposed matrix, \f$ \hat{M}^{\dagger}\f$.
//! @tparam T Value type of CRS matrix.
//! @param matrix CRS object \f$ \hat{M}\f$.
//! @return CRS object \f$ \hat{M}^{\dagger} \f$.
template <typename T>
CRS<T> GenerateTransposedMatrix(const CRS<T> &matrix) {
   CRS<T> matrix_out(matrix.col_dim, matrix.row_dim, matrix.tag);
   
   std::vector<std::int64_t> row_count(matrix.row_dim);
   for (std::int64_t i = 0; i < matrix.col_dim; ++i) {
      for (std::int64_t j = 0; j < matrix.row_dim; ++j) {
         const std::int64_t row = matrix.row[j] + row_count[j];
         if (row < matrix.row[j + 1] && matrix.col[row] == i) {
            matrix_out.val.push_back(matrix.val[row]);
            matrix_out.col.push_back(j);
            row_count[j]++;
         }
      }
      matrix_out.row[i + 1] = matrix_out.col.size();
   }
   
   return matrix_out;
}

//! @brief Calculate scalar BraketVector product, \f$ c\hat{M}\f$.
//! @tparam T1 Value type of coeff.
//! @tparam T2 Value type of matrix.
//! @param coeff The coefficient \f$ c\f$
//! @param matrix CRS object \f$ \hat{M}\f$.
//! @return CRS object \f$ c\hat{M} \f$.
template <typename T1, typename T2>
auto CalculateScalarMatrixProduct(const T1 coeff, const CRS<T2> &matrix)
-> CRS<decltype(std::declval<T1>() * std::declval<T2>())> {
   using T1T2 = decltype(std::declval<T1>() * std::declval<T2>());
   CRS<T1T2> out(matrix.row_dim, matrix.col_dim, matrix.tag);
   
   if (coeff == T1{0}) {
      return out;
   }
   
   out.col.resize(matrix.col.size());
   out.val.resize(matrix.val.size());
   
#pragma omp parallel for
   for (std::size_t i = 0; i < matrix.col.size(); ++i) {
      out.col[i] = matrix.col[i];
      out.val[i] = coeff * matrix.val[i];
   }
   
#pragma omp parallel for
   for (std::int64_t i = 1; i <= matrix.row_dim; ++i) {
      out.row[i] = matrix.row[i];
   }
   return out;
}

//! @brief Calculate matrix product, \f$ c_{1}\hat{M}_{1}\hat{M}_{2}\f$.
//! @tparam T1 Value type of coeff_1.
//! @tparam T2 Value type of matrix_1.
//! @tparam T3 Value type of matrix_2.
//! @param coeff_1 The coefficient \f$ c_1 \f$.
//! @param matrix_1 The CRS object of the left-hand side, \f$ \hat{M}_{1} \f$.
//! @param matrix_2 The CRS object of the right-hand side, \f$ \hat{M}_{2} \f$.
//! @return CRS object, \f$ c_1\hat{M}_{1}\hat{M}_{2}\f$
template <typename T1, typename T2, typename T3>
auto CalculateMatrixMatrixProduct(const T1 coeff_1, const CRS<T2> &matrix_1, const CRS<T3> &matrix_2)
-> CRS<decltype(std::declval<T1>() * std::declval<T2>() * std::declval<T3>())> {
   if (matrix_1.col_dim != matrix_2.row_dim) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "Matrix product cannot be defined" << std::endl;
      ss << "matrix_1.col_dim = " << matrix_1.col_dim << ", matrix_2.row_dim = " << matrix_2.row_dim << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   using T1T2 = decltype(std::declval<T1>() * std::declval<T2>());
   using T1T2T3 = decltype(std::declval<T1>() * std::declval<T2>() * std::declval<T3>());
   
   CRS<T1T2T3> matrix_out(matrix_1.row_dim, matrix_2.col_dim);
   
   std::vector<T1T2> temp_v1(matrix_1.col_dim, 0.0);
   std::vector<T1T2T3> temp_v2(matrix_2.col_dim, 0.0);
   
   for (std::int64_t i = 0; i < matrix_1.row_dim; ++i) {
      for (std::int64_t j = matrix_1.row[i]; j < matrix_1.row[i + 1]; ++j) {
         temp_v1[matrix_1.col[j]] = coeff_1 * matrix_1.val[j];
      }
      for (std::int64_t j = 0; j < matrix_1.col_dim; ++j) {
         for (std::int64_t k = matrix_2.row[j]; k < matrix_2.row[j + 1]; ++k) {
            temp_v2[matrix_2.col[k]] += temp_v1[j] * matrix_2.val[k];
         }
      }
      for (std::int64_t j = 0; j < matrix_2.col_dim; ++j) {
         if (std::abs(temp_v2[j]) > 0.0) {
            matrix_out.val.push_back(temp_v2[j]);
            matrix_out.col.push_back(j);
         }
      }
      
      matrix_out.row[i + 1] = matrix_out.col.size();
      
      for (std::int64_t j = matrix_1.row[i]; j < matrix_1.row[i + 1]; ++j) {
         temp_v1[matrix_1.col[j]] = 0.0;
      }
      for (std::int64_t j = matrix_out.row[i]; j < matrix_out.row[i + 1]; ++j) {
         temp_v2[matrix_out.col[j]] = 0.0;
      }
   }
   
   if (matrix_1.tag == CRSTag::NONE) {
      matrix_out.tag = matrix_2.tag;
   } else if (matrix_1.tag == CRSTag::FERMION) {
      if (matrix_2.tag == CRSTag::NONE || matrix_2.tag == CRSTag::BOSON) {
         matrix_out.tag = CRSTag::FERMION;
      } else if (matrix_2.tag == CRSTag::FERMION) {
         matrix_out.tag = CRSTag::BOSON;
      } else if (matrix_2.tag == CRSTag::MIX) {
         matrix_out.tag = CRSTag::MIX;
      } else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Unknown CRSTag detected.";
         throw std::runtime_error(ss.str());
      }
   } else if (matrix_1.tag == CRSTag::BOSON) {
      if (matrix_2.tag == CRSTag::FERMION || matrix_2.tag == CRSTag::BOSON || matrix_2.tag == CRSTag::MIX) {
         matrix_out.tag = matrix_2.tag;
      } else if (matrix_2.tag == CRSTag::NONE) {
         matrix_out.tag = CRSTag::BOSON;
      } else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Unknown CRSTag detected.";
         throw std::runtime_error(ss.str());
      }
   } else if (matrix_1.tag == CRSTag::MIX) {
      matrix_out.tag = CRSTag::MIX;
   } else {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "Unknown CRSTag detected.";
      throw std::runtime_error(ss.str());
   }
   
   return matrix_out;
}

//! @brief Calculate matrix summation, \f$ c_{1}\hat{M}_{1} + c_{2}\hat{M}_{2}\f$
//! @tparam T1 Value type of coeff_1.
//! @tparam T2 Value type of matrix_1.
//! @tparam T3 Value type of coeff_2.
//! @tparam T4 Value type of matrix_2.
//! @param coeff_1 The coefficient \f$ c_1 \f$.
//! @param matrix_1 The CRS object of the left-hand side, \f$ \hat{M}_{1} \f$.
//! @param coeff_2 The coefficient \f$ c_2 \f$.
//! @param matrix_2 The CRS object of the right-hand side, \f$ \hat{M}_{2} \f$.
//! @return CRS object, \f$ c_{1}\hat{M}_{1} + c_{2}\hat{M}_{2}\f$
template <typename T1, typename T2, typename T3, typename T4>
auto CalculateMatrixMatrixSum(const T1 coeff_1, const CRS<T2> &matrix_1, const T3 coeff_2, const CRS<T4> &matrix_2)
-> CRS<decltype(std::declval<T1>() * std::declval<T2>() + std::declval<T3>() * std::declval<T4>())> {
   if (matrix_1.row_dim != matrix_2.row_dim || matrix_1.col_dim != matrix_2.col_dim) {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "The summation of the matrices cannot be defined." << std::endl;
      ss << "matrix_1.row_dim = " << matrix_1.row_dim << ", matrix_1.col_dim = " << matrix_1.col_dim << std::endl;
      ss << "matrix_2.row_dim = " << matrix_2.row_dim << ", matrix_2.col_dim = " << matrix_2.col_dim << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   using T1T2T3T4 = decltype(std::declval<T1>() * std::declval<T2>() + std::declval<T3>() * std::declval<T4>());
   
   CRS<T1T2T3T4> matrix_out(matrix_1.row_dim, matrix_1.col_dim);
   
   for (std::int64_t i = 0; i < matrix_1.row_dim; ++i) {
      int check = 0;
      std::int64_t count_1 = 0;
      std::int64_t count_2 = 0;
      
      const std::int64_t row_lower_1 = matrix_1.row[i];
      const std::int64_t row_upper_1 = matrix_1.row[i + 1];
      const std::int64_t row_lower_2 = matrix_2.row[i];
      const std::int64_t row_upper_2 = matrix_2.row[i + 1];
      
      const std::int64_t m1_count = row_upper_1 - row_lower_1;
      const std::int64_t m2_count = row_upper_2 - row_lower_2;
      
      if (m1_count != 0 && m2_count == 0) {
         for (std::int64_t j = row_lower_1; j < row_upper_1; ++j) {
            matrix_out.val.push_back(coeff_1 * matrix_1.val[j]);
            matrix_out.col.push_back(matrix_1.col[j]);
         }
      }
      else if (m1_count == 0 && m2_count != 0) {
         for (std::int64_t j = row_lower_2; j < row_upper_2; ++j) {
            matrix_out.val.push_back(coeff_2 * matrix_2.val[j]);
            matrix_out.col.push_back(matrix_2.col[j]);
         }
      }
      else if (m1_count != 0 && m2_count != 0) {
         for (std::int64_t j = 0; j < m1_count + m2_count; ++j) {
            if (matrix_1.col[row_lower_1 + count_1] < matrix_2.col[row_lower_2 + count_2]) {
               matrix_out.val.push_back(coeff_1 * matrix_1.val[row_lower_1 + count_1]);
               matrix_out.col.push_back(matrix_1.col[row_lower_1 + count_1]);
               count_1++;
               if (row_lower_1 + count_1 == row_upper_1) {
                  check = 1;
                  break;
               }
            }
            else if (matrix_1.col[row_lower_1 + count_1] == matrix_2.col[row_lower_2 + count_2]) {
               const auto val =
               coeff_1 * matrix_1.val[row_lower_1 + count_1] + coeff_2 * matrix_2.val[row_lower_2 + count_2];
               if (std::abs(val) > 0.0) {
                  matrix_out.val.push_back(val);
                  matrix_out.col.push_back(matrix_1.col[row_lower_1 + count_1]);
                  count_1++;
                  count_2++;
                  const std::int64_t temp_count_1 = row_lower_1 + count_1;
                  const std::int64_t temp_count_2 = row_lower_2 + count_2;
                  if (temp_count_1 == row_upper_1 && temp_count_2 < row_upper_2) {
                     check = 1;
                     break;
                  }
                  else if (temp_count_1 < row_upper_1 && temp_count_2 == row_upper_2) {
                     check = 2;
                     break;
                  }
                  else if (temp_count_1 == row_upper_1 && temp_count_2 == row_upper_2) {
                     check = 0;
                     break;
                  }
               }
               else {
                  count_1++;
                  count_2++;
                  const std::int64_t temp_count_1 = row_lower_1 + count_1;
                  const std::int64_t temp_count_2 = row_lower_2 + count_2;
                  if (temp_count_1 == row_upper_1 && temp_count_2 < row_upper_2) {
                     check = 1;
                     break;
                  }
                  else if (temp_count_1 < row_upper_1 && temp_count_2 == row_upper_2) {
                     check = 2;
                     break;
                  }
                  else if (temp_count_1 == row_upper_1 && temp_count_2 == row_upper_2) {
                     check = 0;
                     break;
                  }
               }
            }
            else {
               matrix_out.val.push_back(coeff_2 * matrix_2.val[row_lower_2 + count_2]);
               matrix_out.col.push_back(matrix_2.col[row_lower_2 + count_2]);
               count_2++;
               if (row_lower_2 + count_2 == row_upper_2) {
                  check = 2;
                  break;
               }
            }
         }
         if (check == 1) {
            for (std::int64_t j = row_lower_2 + count_2; j < row_upper_2; ++j) {
               matrix_out.val.push_back(coeff_2 * matrix_2.val[j]);
               matrix_out.col.push_back(matrix_2.col[j]);
            }
         }
         else if (check == 2) {
            for (std::int64_t j = row_lower_1 + count_1; j < row_upper_1; ++j) {
               matrix_out.val.push_back(coeff_1 * matrix_1.val[j]);
               matrix_out.col.push_back(matrix_1.col[j]);
            }
         }
      }
      matrix_out.row[i + 1] = matrix_out.col.size();
   }
   
   if (matrix_1.tag == CRSTag::NONE) {
      matrix_out.tag = matrix_2.tag;
   }
   else if (matrix_1.tag == CRSTag::FERMION) {
      if (matrix_2.tag == CRSTag::NONE || matrix_2.tag == CRSTag::FERMION) {
         matrix_out.tag = CRSTag::FERMION;
      }
      else if (matrix_2.tag == CRSTag::BOSON || matrix_2.tag == CRSTag::MIX) {
         matrix_out.tag = CRSTag::MIX;
      }
      else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Unknown CRSTag detected.";
         throw std::runtime_error(ss.str());
      }
   }
   else if (matrix_1.tag == CRSTag::BOSON) {
      if (matrix_2.tag == CRSTag::NONE || matrix_2.tag == CRSTag::BOSON) {
         matrix_out.tag = CRSTag::BOSON;
      }
      else if (matrix_2.tag == CRSTag::FERMION || matrix_2.tag == CRSTag::MIX) {
         matrix_out.tag = CRSTag::MIX;
      }
      else {
         std::stringstream ss;
         ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
         ss << "Unknown CRSTag detected.";
         throw std::runtime_error(ss.str());
      }
   }
   else if (matrix_1.tag == CRSTag::MIX) {
      matrix_out.tag = CRSTag::MIX;
   }
   else {
      std::stringstream ss;
      ss << "Error at " << __LINE__ << " in " << __func__ << " in " << __FILE__ << std::endl;
      ss << "Unknown CRSTag detected.";
      throw std::runtime_error(ss.str());
   }
   return matrix_out;
}

} // namespace blas
} // namespace compnal

#endif /* COMPNAL_BLAS_COMPRESSED_ROW_STORAGE_HPP_ */
