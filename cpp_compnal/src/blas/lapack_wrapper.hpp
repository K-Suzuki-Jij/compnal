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
//  lapack_wrapper.hpp
//  compnal
//
//  Created by kohei on 2023/01/03.
//  
//

#ifndef COMPNAL_BLAS_LAPACK_WRAPPER_HPP_
#define COMPNAL_BLAS_LAPACK_WRAPPER_HPP_

#include "compressed_row_storage.hpp"

namespace compnal {
namespace blas {

// Float
extern "C" {
void ssyev_(const char &JOBZ, const char &UPLO, const std::int32_t &N, float *A, const std::int32_t &LDA, float *W, float *work,
            const std::int32_t &Lwork, std::int32_t &INFO);
};

extern "C" {
void sstev_(const char &JOBZ, const std::int32_t &N, float *D, float *E, float *Z, const std::int32_t &LDZ, float *WORK, std::int32_t &INFO);
};

extern "C" {
void sspgv_(const std::int32_t &ITYPE, const char &JOBZ, const char &UPLO, const std::int32_t &N, float *AP, float *BP, float *W,
            float *Z, const std::int32_t &LDZ, float *WORK, std::int32_t &INFO);
};

// Double
extern "C" {
void dsyev_(const char &JOBZ, const char &UPLO, const std::int32_t &N, double *A, const std::int32_t &LDA, double *W, double *work,
            const std::int32_t &Lwork, std::int32_t &INFO);
};

extern "C" {
void dstev_(const char &JOBZ, const std::int32_t &N, double *D, double *E, double *Z, const std::int32_t &LDZ, double *WORK, std::int32_t &INFO);
};

extern "C" {
void dspgv_(const std::int32_t &ITYPE, const char &JOBZ, const char &UPLO, const std::int32_t &N, double *AP, double *BP, double *W,
            double *Z, const std::int32_t &LDZ, double *WORK, std::int32_t &INFO);
};

template<typename RealType, class SPMType>
void LapackSYEV(RealType *gs_value,
                std::vector<RealType> *gs_vector,
                const std::int32_t target_level,
                const SPMType &matrix_in
                ) {
   
   if (matrix_in.row_dim != matrix_in.col_dim || matrix_in.row_dim < 1 || matrix_in.col_dim < 1) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "The input matrix is not a square one" << std::endl;
      ss << "row=" << matrix_in.row_dim << ", col=" << matrix_in.col_dim << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   if (target_level < 0) {
      std::stringstream ss;
      ss << "Invalid target_level" << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   const std::int32_t dim = static_cast<std::int32_t>(matrix_in.row_dim);
   std::int32_t info;
   std::unique_ptr<RealType[]> matrix_array = matrix_in.ToArray();
   std::unique_ptr<RealType[]> val_array = std::make_unique<RealType[]>(dim);
   std::unique_ptr<RealType[]> work = std::make_unique<RealType[]>(3*dim);
      
   if constexpr (std::is_same<RealType, double>::value) {
      dsyev_('V', 'L', dim, matrix_array.get(), dim, val_array.get(), work.get(), 3*dim, info);
   }
   else if constexpr (std::is_same<RealType, float>::value) {
      ssyev_('V', 'L', dim, matrix_array.get(), dim, val_array.get(), work.get(), 3*dim, info);
   }
   else {
      static_assert([]{return false;}(), "RealType must be float or double");
   }
   
   gs_vector->resize(dim);
   
   for (std::int32_t i = 0; i < dim; ++i) {
      (*gs_vector)[i] = matrix_array[target_level*dim + i];
   }
   
   *gs_value = val_array[target_level];
}

template<typename RealType>
void LapackSTEV(RealType *gs_value,
                std::vector<RealType> *gs_vector,
                const std::vector<RealType> &diag,
                const std::vector<RealType> &off_diag
                ) {
   
   if (off_diag.size() + 1 != diag.size()) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      ss << "diag size=" << diag.size() << ", off_diag size=" << off_diag.size() << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   std::int32_t dim = static_cast<std::int32_t>(diag.size());
   
   std::int32_t info;
   std::unique_ptr<RealType[]> lap_d = std::make_unique<RealType[]>(dim);
   std::unique_ptr<RealType[]> lap_e = std::make_unique<RealType[]>(dim - 1);
   std::unique_ptr<RealType[]> lap_vec = std::make_unique<RealType[]>(dim*dim);
   std::unique_ptr<RealType[]> lap_work = std::make_unique<RealType[]>(2*dim);
      
   for (std::int32_t i = 0; i < dim; i++) {
      lap_d[i] = diag[i];
   }
   
   for (std::int32_t i = 0; i < dim - 1; i++) {
      lap_e[i] = off_diag[i];
   }
   
   if constexpr (std::is_same<RealType, double>::value) {
      dstev_('V', dim, lap_d.get(), lap_e.get(), lap_vec.get(), dim, lap_work.get(), info);
   }
   else if constexpr (std::is_same<RealType, float>::value) {
      sstev_('V', dim, lap_d.get(), lap_e.get(), lap_vec.get(), dim, lap_work.get(), info);
   }
   else {
      static_assert([]{return false;}(), "RealType must be float or double");
   }
      
   gs_vector->resize(dim);
   
   for (std::int32_t i = 0; i < dim; ++i) {
      (*gs_vector)[i] = lap_vec[i];
   }
   
   *gs_value = lap_d[0];
}

template<typename RealType>
void LapackDspgv(const std::int32_t i_type,
                 const std::int32_t dim,
                 const std::vector<RealType> &mat_a,
                 const std::vector<RealType> &mat_b,
                 std::vector<RealType> *eigenvalues,
                 std::vector<RealType> *eigenvectors) {
   
   const std::int32_t size = dim*(dim + 1)/2;
   
   if (static_cast<std::int32_t>(mat_a.size()) < size || static_cast<std::int32_t>(mat_b.size()) < size) {
      std::stringstream ss;
      ss << "Error in " << __func__ << std::endl;
      throw std::runtime_error(ss.str());
   }
   
   std::unique_ptr<RealType[]> lap_ap = std::make_unique<RealType[]>(size);
   std::unique_ptr<RealType[]> lap_bp = std::make_unique<RealType[]>(size);
   std::unique_ptr<RealType[]> lap_w = std::make_unique<RealType[]>(dim);
   std::unique_ptr<RealType[]> lap_z = std::make_unique<RealType[]>(dim*dim);
   std::unique_ptr<RealType[]> lap_work = std::make_unique<RealType[]>(3*dim);
   std::int32_t lap_ldz = dim;
   std::int32_t lap_info;
   
   for (std::int32_t i = 0; i < size; ++i) {
      lap_ap[i] = mat_a[i];
      lap_bp[i] = mat_b[i];
   }
   
   if constexpr (std::is_same<RealType, double>::value) {
      dspgv_(i_type, 'V', 'U', dim, lap_ap.get(), lap_bp.get(), lap_w.get(), lap_z.get(), lap_ldz, lap_work.get(), lap_info);
   }
   else if constexpr (std::is_same<RealType, float>::value) {
      sspgv_(i_type, 'V', 'U', dim, lap_ap.get(), lap_bp.get(), lap_w.get(), lap_z.get(), lap_ldz, lap_work.get(), lap_info);
   }
   else {
      static_assert([]{return false;}(), "RealType must be float or double");
   }
   
   eigenvalues->resize(dim);
   eigenvectors->resize(dim*dim);
   
   for (std::int32_t i = 0; i < dim; ++i) {
      (*eigenvalues)[i] = lap_w[i];
      for (std::int32_t j = 0; j < dim; ++j) {
         (*eigenvectors)[i*dim + j] = lap_z[i][j];
      }
   }
}


} // namespace blas
} // namespace compnal

#endif /* COMPNAL_BLAS_LAPACK_WRAPPER_HPP_ */
