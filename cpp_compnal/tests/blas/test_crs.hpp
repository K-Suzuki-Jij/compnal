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
//  test_crs.hpp
//  compnal
//
//  Created by kohei on 2022/10/11.
//  
//

#ifndef COMPNAL_TEST_BLAS_CRS_HPP_
#define COMPNAL_TEST_BLAS_CRS_HPP_

#include <gtest/gtest.h>
#include "../../src/blas/all.hpp"

namespace compnal {
namespace test {

TEST(CRS, Constructors) {
   EXPECT_NO_THROW(blas::CRS<double>());
   EXPECT_NO_THROW(blas::CRS<double>(3, 4));
   EXPECT_NO_THROW(blas::CRS<double>(3, 4, blas::CRSTag::FERMION));
   EXPECT_NO_THROW(blas::CRS<double>({{1, 2.0}, {3.1}, {-1}}));
   EXPECT_NO_THROW(blas::CRS<double>({{1, 2.0}, {3.1}, {-1}}, blas::CRSTag::FERMION));
   EXPECT_NO_THROW(blas::CRS<double>(std::vector<std::vector<int>>{{1, 2, 3}, {-1, -2}}));
   EXPECT_NO_THROW(blas::CRS<int>());
   EXPECT_NO_THROW(blas::CRS<int>(3, 4));
   EXPECT_NO_THROW(blas::CRS<int>(3, 4, blas::CRSTag::FERMION));
   EXPECT_NO_THROW(blas::CRS<int>({{1, 2}, {3}, {-1}}));
   EXPECT_NO_THROW(blas::CRS<int>({{1, 2}, {3}, {-1}}, blas::CRSTag::FERMION));
   EXPECT_NO_THROW(blas::CRS<int>(std::vector<std::vector<int>>{{1, 2, 3}, {-1, -2}}));
   EXPECT_EQ(blas::CRS<int>({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}), blas::CRS<int>(3, 3));
}

TEST(CRS, Free) {
   blas::CRS<int> m({{1, 2}, {3}, {-1}}, blas::CRSTag::FERMION);
   m.name = "Matrix";
   m.Free();
   EXPECT_EQ(m, blas::CRS<int>());
}

TEST(CRS, Assign) {
   blas::CRS<int> m_int({{1, 2}, {3}, {-1}}, blas::CRSTag::BOSON);

   blas::CRS<double> m_d_1 = m_int;
   EXPECT_EQ(m_d_1, m_int);

   blas::CRS<double> m_d_2;
   m_d_2 = m_int;
   EXPECT_EQ(m_d_2, m_int);

   blas::CRS<double> m_d_3;
   m_d_3.Assign(m_int);
   EXPECT_EQ(m_d_3, m_int);
}

TEST(CRS, MultiplyByScalar) {
   blas::CRS<double> m_d({{1.0, 2.0}, {3.0}, {-2.0}});
   m_d.MultiplyByScalar(2);
   EXPECT_EQ(m_d, blas::CRS<double>({{2.0, 4.0}, {6.0}, {-4.0}}));

   blas::CRS<int> m_int({{1, 3}, {3}, {-1}});
   m_int.MultiplyByScalar(2);
   EXPECT_EQ(m_int, blas::CRS<int>({{2, 6}, {6}, {-2}}));
}

TEST(CRS, SortCol) {
   blas::CRS<int> m_d(3, 3);
   m_d.col.resize(8);
   m_d.val.resize(8);
   m_d.row[1] = 2;
   m_d.row[2] = 5;
   m_d.row[3] = 8;
   m_d.col[0] = 2;
   m_d.col[1] = 1;
   m_d.col[2] = 1;
   m_d.col[3] = 2;
   m_d.col[4] = 0;
   m_d.col[5] = 2;
   m_d.col[6] = 0;
   m_d.col[7] = 1;
   m_d.val[0] = 2;
   m_d.val[1] = 1;
   m_d.val[2] = 3;
   m_d.val[3] = 4;
   m_d.val[4] = 2;
   m_d.val[5] = 7;
   m_d.val[6] = 5;
   m_d.val[7] = 6;
   EXPECT_FALSE(m_d == blas::CRS<int>({{0, 1, 2}, {2, 3, 4}, {5, 6, 7}}));
   m_d.SortCol();
   EXPECT_TRUE(m_d == blas::CRS<int>({{0, 1, 2}, {2, 3, 4}, {5, 6, 7}}));
}

TEST(CRS, CheckSymmetric) {
   blas::CRS<double> m_d_1({{1.0, 2.0}, {3.0, -2.0}});
   EXPECT_FALSE(m_d_1.CheckSymmetric());

   blas::CRS<double> m_d_2({{1.0, 2.0}, {-2.0}, {3.0, -2.0}});
   EXPECT_THROW(m_d_2.CheckSymmetric(), std::runtime_error);

   blas::CRS<double> m_d_3({{1.0, 2.0}, {2.0, 1.0}});
   EXPECT_TRUE(m_d_3.CheckSymmetric());
}

TEST(CRS, UnaryOperators) {
   blas::CRS<double> m_d({{1.0, 2.0}, {-2.0}, {3.0, -2.0}});
   EXPECT_EQ(+m_d, blas::CRS<double>({{1.0, 2.0}, {-2.0}, {3.0, -2.0}}));
   EXPECT_EQ(-m_d, blas::CRS<double>({{-1.0, -2.0}, {2.0}, {-3.0, 2.0}}));
}

TEST(CRS, CompoundAssignmentOperators) {
   blas::CRS<int> m_d_1({{+1, +2}, {+2, +1}});
   blas::CRS<int> m_d_2({{-3, +1}, {-2, -2}});
   blas::CRS<int> m_d_3 = m_d_1 + m_d_2;
   m_d_1 += m_d_2;
   EXPECT_EQ(m_d_1, m_d_3);
   m_d_1 -= m_d_2;
   EXPECT_EQ(m_d_1, m_d_3 - m_d_2);
   m_d_1 *= m_d_2;
   EXPECT_EQ(m_d_1, blas::CRS<int>({{+1, +2}, {+2, +1}}) * m_d_2);
}

TEST(CRS, CalculateMatrixMatrixSum) {
   blas::CRS<int> m_1(2, 2);
   blas::CRS<int> m_2(2, 2);
   blas::CRS<int> m_3(2, 2);

   // CRSTag
   //------------------------------
   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   //------------------------------
   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   //------------------------------
   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   //------------------------------
   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixSum(1, m_1, 1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::NONE));

   // Check Sum
   m_1 = blas::CRS<int>({{1, 2}, {3, 4}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{5, 6}, {7, 8}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixSum(2, m_1, 3, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{2 + 15, 4 + 18}, {6 + 21, 8 + 24}}));

   m_1 = blas::CRS<int>({{0, 0}, {0, 0}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{1, 2}, {3, 4}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixSum(2, m_1, 3, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{3, 6}, {9, 12}}));

   m_1 = blas::CRS<int>({{1, 2}, {3, 4}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{0, 0}, {0, 0}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixSum(2, m_1, 3, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{2, 4}, {6, 8}}));

   m_1 = blas::CRS<int>({{0, 0}, {5, 0}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{1, 2}, {3, 4}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixSum(2, m_1, 3, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{3, 6}, {9 + 10, 12}}));

   m_1 = blas::CRS<int>({{1, 2}, {3, 4}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{0, 5}, {0, 0}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixSum(2, m_1, 3, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{2, 4 + 15}, {6, 8}}));

   m_1 = blas::CRS<int>({{1, 2, 3, 4}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{0, 5}, {0, 0}}, blas::CRSTag::NONE);
   EXPECT_THROW(blas::CalculateMatrixMatrixSum(2, m_1, 3, m_2), std::runtime_error);
}

TEST(CRS, CalculateMatrixMatrixProduct) {
   blas::CRS<int> m_1(2, 2);
   blas::CRS<int> m_2(2, 2);
   blas::CRS<int> m_3(2, 2);

   // CRSTag
   //------------------------------
   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::FERMION;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   //------------------------------
   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::BOSON;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   //------------------------------
   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::MIX;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   //------------------------------
   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::FERMION;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::FERMION));

   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::BOSON;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::BOSON));

   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::MIX;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::MIX));

   m_1.tag = blas::CRSTag::NONE;
   m_2.tag = blas::CRSTag::NONE;
   m_3 = blas::CalculateMatrixMatrixProduct(1, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>(2, 2, blas::CRSTag::NONE));

   // Check Prod
   m_1 = blas::CRS<int>({{1, 2, 3}, {4, 5, 6}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{7, 8}, {9, 10}, {11, 12}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixProduct(2, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{58 * 2, 64 * 2}, {139 * 2, 154 * 2}}));

   m_1 = blas::CRS<int>({{1, 2, 0}, {4, 0, 0}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{7, 8}, {9, 10}, {11, 12}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixProduct(2, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{25 * 2, 28 * 2}, {28 * 2, 32 * 2}}));

   m_1 = blas::CRS<int>({{1, 2, 3}, {4, 5, 6}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{7, 8}, {0, 0}, {11, 0}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixProduct(2, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{40 * 2, 8 * 2}, {94 * 2, 32 * 2}}));

   m_1 = blas::CRS<int>({{0, 0, 0}, {0, 0, 0}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{7, 8}, {9, 10}, {11, 12}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixProduct(2, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{0, 0}, {0, 0}}));

   m_1 = blas::CRS<int>({{1, 2, 3}, {4, 5, 6}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{0, 0}, {0, 0}, {0, 0}}, blas::CRSTag::NONE);
   m_3 = blas::CalculateMatrixMatrixProduct(2, m_1, m_2);
   EXPECT_EQ(m_3, blas::CRS<int>({{0, 0}, {0, 0}}));

   m_1 = blas::CRS<int>({{1, 2, 3, 4}}, blas::CRSTag::NONE);
   m_2 = blas::CRS<int>({{0, 5}, {0, 0}}, blas::CRSTag::NONE);
   EXPECT_THROW(blas::CalculateMatrixMatrixProduct(2, m_1, m_2), std::runtime_error);
}

TEST(CRS, CalculateScalarMatrixProduct) {
   blas::CRS<int> m_1({{1, 2}, {3, 4}});
   auto m_2 = blas::CalculateScalarMatrixProduct(2, m_1);
   EXPECT_EQ(m_2, blas::CRS<int>({{2, 4}, {6, 8}}));

   m_2 = blas::CalculateScalarMatrixProduct(0, m_1);
   EXPECT_EQ(m_2, blas::CRS<int>({{0, 0}, {0, 0}}));
}

TEST(CRS, CalculateTransposedMatrix) {
   blas::CRS<int> m_1({{1, 2, 3}, {4, 5, 6}}, blas::CRSTag::NONE);
   auto m_2 = blas::GenerateTransposedMatrix(m_1);
   EXPECT_EQ(m_2, blas::CRS<int>({{1, 4}, {2, 5}, {3, 6}}));
}

TEST(CRS, Addition) {
   auto expect_crs_eq = [](const auto &lhs, const auto &rhs, const auto &prec, const auto &line) {
      const bool c1 = std::is_integral<typename std::remove_cvref<decltype(lhs)>::type::ValueType>::value;
      const bool c2 = std::is_integral<typename std::remove_cvref<decltype(rhs)>::type::ValueType>::value;
      if (c1 && c2) {
         EXPECT_EQ(lhs, rhs);
      } else {
         EXPECT_EQ(lhs.row_dim, rhs.row_dim);
         EXPECT_EQ(lhs.col_dim, rhs.col_dim);
         EXPECT_EQ(lhs.row.size(), rhs.row.size());
         EXPECT_EQ(lhs.col.size(), rhs.col.size());
         EXPECT_EQ(lhs.val.size(), rhs.val.size());
         EXPECT_EQ(lhs.tag, rhs.tag);
         EXPECT_EQ(lhs.name, rhs.name);
         for (std::size_t i = 0; i < lhs.row.size(); ++i) {
            EXPECT_EQ(lhs.row.at(i), rhs.row.at(i));
         }
         for (std::size_t i = 0; i < lhs.col.size(); ++i) {
            EXPECT_EQ(lhs.col.at(i), rhs.col.at(i));
         }
         for (std::size_t i = 0; i < lhs.val.size(); ++i) {
            const auto lhs_val = std::abs(lhs.val.at(i) - rhs.val.at(i));
            const auto rhs_val = 4 * prec;
            EXPECT_TRUE(lhs_val <= rhs_val) << lhs << " !<= " << rhs << " from " << line;
         }
      }
   };
   using blas::CRS;
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<long double>({{1.4L}}), CRS<long double>({{2.7L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<double>({{1.4}}), CRS<double>({{2.7}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<float>({{1.4F}}), CRS<float>({{2.7F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<long long>({{1LL}}), CRS<long double>({{2.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<long>({{1L}}), CRS<long double>({{2.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<int>({{1}}), CRS<long double>({{2.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<short>({{1}}), CRS<long double>({{2.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) + CRS<char>({{1}}), CRS<long double>({{2.3L}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<double>({{1.3}}) + CRS<long double>({{1.4L}}), CRS<double>({{2.7}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<double>({{1.4}}), CRS<double>({{2.7}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<float>({{1.4F}}), CRS<float>({{2.7F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<long long>({{1LL}}), CRS<double>({{2.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<long>({{1L}}), CRS<double>({{2.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<int>({{1}}), CRS<double>({{2.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<short>({{1}}), CRS<double>({{2.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) + CRS<char>({{1}}), CRS<double>({{2.3}}), DBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<long double>({{1.4L}}), CRS<float>({{2.7F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<double>({{1.4}}), CRS<float>({{2.7F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<float>({{1.4F}}), CRS<float>({{2.7F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<long long>({{1LL}}), CRS<float>({{2.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<long>({{1L}}), CRS<float>({{2.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<int>({{1}}), CRS<float>({{2.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<short>({{1}}), CRS<float>({{2.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) + CRS<char>({{1}}), CRS<float>({{2.3F}}), FLT_EPSILON, __LINE__);

   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<long double>({{1.4L}}), CRS<long double>({{2.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<double>({{1.4}}), CRS<double>({{2.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<float>({{1.4F}}), CRS<float>({{2.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<long long>({{1LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<long>({{1L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<int>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<short>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) + CRS<char>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<long>({{1L}}) + CRS<long double>({{1.4L}}), CRS<long double>({{2.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<double>({{1.4}}), CRS<double>({{2.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<float>({{1.4F}}), CRS<float>({{2.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<long long>({{1LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<long>({{1L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<int>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<short>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) + CRS<char>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<int>({{1}}) + CRS<long double>({{1.4L}}), CRS<long double>({{2.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<double>({{1.4}}), CRS<double>({{2.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<float>({{1.4F}}), CRS<float>({{2.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<long long>({{1LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<long>({{1L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<int>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<short>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) + CRS<char>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<short>({{1}}) + CRS<long double>({{1.4L}}), CRS<long double>({{2.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<double>({{1.4}}), CRS<double>({{2.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<float>({{1.4F}}), CRS<float>({{2.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<long long>({{1LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<long>({{1L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<int>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<short>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) + CRS<char>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<char>({{1}}) + CRS<long double>({{1.4L}}), CRS<long double>({{2.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<double>({{1.4}}), CRS<double>({{2.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<float>({{1.4F}}), CRS<float>({{2.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<long long>({{1LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<long>({{1L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<int>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<short>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) + CRS<char>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);
}

TEST(CRS, Subtraction) {
   auto expect_crs_eq = [](const auto &lhs, const auto &rhs, const auto &prec, const auto &line) {
      const bool c1 = std::is_integral<typename std::remove_cvref<decltype(lhs)>::type::ValueType>::value;
      const bool c2 = std::is_integral<typename std::remove_cvref<decltype(rhs)>::type::ValueType>::value;
      if (c1 && c2) {
         EXPECT_EQ(lhs, rhs);
      } else {
         EXPECT_EQ(lhs.row_dim, rhs.row_dim);
         EXPECT_EQ(lhs.col_dim, rhs.col_dim);
         EXPECT_EQ(lhs.row.size(), rhs.row.size());
         EXPECT_EQ(lhs.col.size(), rhs.col.size());
         EXPECT_EQ(lhs.val.size(), rhs.val.size());
         EXPECT_EQ(lhs.tag, rhs.tag);
         EXPECT_EQ(lhs.name, rhs.name);
         for (std::size_t i = 0; i < lhs.row.size(); ++i) {
            EXPECT_EQ(lhs.row.at(i), rhs.row.at(i));
         }
         for (std::size_t i = 0; i < lhs.col.size(); ++i) {
            EXPECT_EQ(lhs.col.at(i), rhs.col.at(i));
         }
         for (std::size_t i = 0; i < lhs.val.size(); ++i) {
            const auto lhs_val = std::abs(lhs.val.at(i) - rhs.val.at(i));
            const auto rhs_val = 4 * prec;
            EXPECT_TRUE(lhs_val <= rhs_val) << lhs << " !<= " << rhs << " from " << line;
         }
      }
   };
   using blas::CRS;
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<long double>({{1.4L}}), CRS<long double>({{-0.1L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<double>({{1.4}}), CRS<double>({{-0.1}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.1F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<long long>({{1LL}}), CRS<long double>({{+0.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<long>({{1L}}), CRS<long double>({{+0.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<int>({{1}}), CRS<long double>({{+0.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<short>({{1}}), CRS<long double>({{+0.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) - CRS<char>({{1}}), CRS<long double>({{+0.3L}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<double>({{1.3}}) - CRS<long double>({{1.4L}}), CRS<double>({{-0.1}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<double>({{1.4}}), CRS<double>({{-0.1}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.1F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<long long>({{1LL}}), CRS<double>({{+0.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<long>({{1L}}), CRS<double>({{+0.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<int>({{1}}), CRS<double>({{+0.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<short>({{1}}), CRS<double>({{+0.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) - CRS<char>({{1}}), CRS<double>({{+0.3}}), DBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<long double>({{1.4L}}), CRS<float>({{-0.1F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<double>({{1.4}}), CRS<float>({{-0.1F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.1F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<long long>({{1LL}}), CRS<float>({{+0.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<long>({{1L}}), CRS<float>({{+0.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<int>({{1}}), CRS<float>({{+0.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<short>({{1}}), CRS<float>({{+0.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) - CRS<char>({{1}}), CRS<float>({{+0.3F}}), FLT_EPSILON, __LINE__);

   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<long double>({{1.4L}}), CRS<long double>({{-0.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<double>({{1.4}}), CRS<double>({{-0.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<long long>({{2LL}}), CRS<long long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<long>({{2L}}), CRS<long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<int>({{2}}), CRS<int>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<short>({{2}}), CRS<short>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) - CRS<char>({{2}}), CRS<char>({{-1LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<long>({{1L}}) - CRS<long double>({{1.4L}}), CRS<long double>({{-0.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<double>({{1.4}}), CRS<double>({{-0.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<long long>({{2LL}}), CRS<long long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<long>({{2L}}), CRS<long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<int>({{2}}), CRS<int>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<short>({{2}}), CRS<short>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) - CRS<char>({{2}}), CRS<char>({{-1LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<int>({{1}}) - CRS<long double>({{1.4L}}), CRS<long double>({{-0.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<double>({{1.4}}), CRS<double>({{-0.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<long long>({{2LL}}), CRS<long long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<long>({{2L}}), CRS<long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<int>({{2}}), CRS<int>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<short>({{2}}), CRS<short>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) - CRS<char>({{2}}), CRS<char>({{-1LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<short>({{1}}) - CRS<long double>({{1.4L}}), CRS<long double>({{-0.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<double>({{1.4}}), CRS<double>({{-0.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<long long>({{2LL}}), CRS<long long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<long>({{2L}}), CRS<long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<int>({{2}}), CRS<int>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<short>({{2}}), CRS<short>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) - CRS<char>({{2}}), CRS<char>({{-1LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<char>({{1}}) - CRS<long double>({{1.4L}}), CRS<long double>({{-0.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<double>({{1.4}}), CRS<double>({{-0.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<float>({{1.4F}}), CRS<float>({{-0.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<long long>({{2LL}}), CRS<long long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<long>({{2L}}), CRS<long>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<int>({{2}}), CRS<int>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<short>({{2}}), CRS<short>({{-1LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) - CRS<char>({{2}}), CRS<char>({{-1LL}}), LDBL_EPSILON, __LINE__);
}

TEST(CRS, MultiplicationMatrixMatrix) {
   auto expect_crs_eq = [](const auto &lhs, const auto &rhs, const auto &prec, const auto &line) {
      const bool c1 = std::is_integral<typename std::remove_cvref<decltype(lhs)>::type::ValueType>::value;
      const bool c2 = std::is_integral<typename std::remove_cvref<decltype(rhs)>::type::ValueType>::value;
      if (c1 && c2) {
         EXPECT_EQ(lhs, rhs);
      } else {
         EXPECT_EQ(lhs.row_dim, rhs.row_dim);
         EXPECT_EQ(lhs.col_dim, rhs.col_dim);
         EXPECT_EQ(lhs.row.size(), rhs.row.size());
         EXPECT_EQ(lhs.col.size(), rhs.col.size());
         EXPECT_EQ(lhs.val.size(), rhs.val.size());
         EXPECT_EQ(lhs.tag, rhs.tag);
         EXPECT_EQ(lhs.name, rhs.name);
         for (std::size_t i = 0; i < lhs.row.size(); ++i) {
            EXPECT_EQ(lhs.row.at(i), rhs.row.at(i));
         }
         for (std::size_t i = 0; i < lhs.col.size(); ++i) {
            EXPECT_EQ(lhs.col.at(i), rhs.col.at(i));
         }
         for (std::size_t i = 0; i < lhs.val.size(); ++i) {
            const auto lhs_val = std::abs(lhs.val.at(i) - rhs.val.at(i));
            const auto rhs_val = 4 * prec;
            EXPECT_TRUE(lhs_val <= rhs_val) << lhs << " !<= " << rhs << " from " << line;
         }
      }
   };
   using blas::CRS;
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<long double>({{1.4L}}), CRS<long double>({{1.82L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<double>({{1.4}}), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<float>({{1.4F}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<long long>({{2LL}}), CRS<long double>({{2.6L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<long>({{2L}}), CRS<long double>({{2.6L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<int>({{2}}), CRS<long double>({{2.6L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<short>({{2}}), CRS<long double>({{2.6L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * CRS<char>({{2}}), CRS<long double>({{2.6L}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<double>({{1.3}}) * CRS<long double>({{1.4L}}), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<double>({{1.4}}), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<float>({{1.4F}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<long long>({{2LL}}), CRS<double>({{2.6}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<long>({{2L}}), CRS<double>({{2.6}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<int>({{2}}), CRS<double>({{2.6}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<short>({{2}}), CRS<double>({{2.6}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * CRS<char>({{2}}), CRS<double>({{2.6}}), DBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<long double>({{1.4L}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<double>({{1.4}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<float>({{1.4F}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<long long>({{2LL}}), CRS<float>({{2.6F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<long>({{2L}}), CRS<float>({{2.6F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<int>({{2}}), CRS<float>({{2.6F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<short>({{2}}), CRS<float>({{2.6F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * CRS<char>({{2}}), CRS<float>({{2.6F}}), FLT_EPSILON, __LINE__);

   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<long double>({{1.4L}}), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<double>({{1.4}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<float>({{1.4F}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<long long>({{2LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<long>({{2L}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<int>({{2}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<short>({{2}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * CRS<char>({{2}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<long>({{1L}}) * CRS<long double>({{1.4L}}), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<double>({{1.4}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<float>({{1.4F}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<long long>({{2LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<long>({{2L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<int>({{2}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<short>({{2}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * CRS<char>({{2}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<int>({{1}}) * CRS<long double>({{1.4L}}), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<double>({{1.4}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<float>({{1.4F}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<long long>({{2LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<long>({{2L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<int>({{2}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<short>({{2}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * CRS<char>({{2}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<short>({{1}}) * CRS<long double>({{1.4L}}), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<double>({{1.4}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<float>({{1.4F}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<long long>({{2LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<long>({{2L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<int>({{2}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<short>({{2}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * CRS<char>({{2}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<char>({{1}}) * CRS<long double>({{1.4L}}), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<double>({{1.4}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<float>({{1.4F}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<long long>({{2LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<long>({{2L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<int>({{2}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<short>({{2}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * CRS<char>({{2}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);
}

TEST(CRS, MultiplicationMatrixScalar) {
   auto expect_crs_eq = [](const auto &lhs, const auto &rhs, const auto &prec, const auto &line) {
      const bool c1 = std::is_integral<typename std::remove_cvref<decltype(lhs)>::type::ValueType>::value;
      const bool c2 = std::is_integral<typename std::remove_cvref<decltype(rhs)>::type::ValueType>::value;
      if (c1 && c2) {
         EXPECT_EQ(lhs, rhs);
      } else {
         EXPECT_EQ(lhs.row_dim, rhs.row_dim);
         EXPECT_EQ(lhs.col_dim, rhs.col_dim);
         EXPECT_EQ(lhs.row.size(), rhs.row.size());
         EXPECT_EQ(lhs.col.size(), rhs.col.size());
         EXPECT_EQ(lhs.val.size(), rhs.val.size());
         EXPECT_EQ(lhs.tag, rhs.tag);
         EXPECT_EQ(lhs.name, rhs.name);
         for (std::size_t i = 0; i < lhs.row.size(); ++i) {
            EXPECT_EQ(lhs.row.at(i), rhs.row.at(i));
         }
         for (std::size_t i = 0; i < lhs.col.size(); ++i) {
            EXPECT_EQ(lhs.col.at(i), rhs.col.at(i));
         }
         for (std::size_t i = 0; i < lhs.val.size(); ++i) {
            const auto lhs_val = std::abs(lhs.val.at(i) - rhs.val.at(i));
            const auto rhs_val = 4 * prec;
            EXPECT_TRUE(lhs_val <= rhs_val) << lhs << " !<= " << rhs << " from " << line;
         }
      }
   };
   using blas::CRS;
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<long double>(1.4L), CRS<long double>({{1.82L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<double>(1.4), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<float>(1.4F), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<long long>(1LL), CRS<long double>({{1.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<long>(1L), CRS<long double>({{1.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<int>(1), CRS<long double>({{1.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<short>(1), CRS<long double>({{1.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long double>({{1.3L}}) * static_cast<char>(1), CRS<long double>({{1.3L}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<long double>(1.4L), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<double>(1.4), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<float>(1.4F), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<long long>(1LL), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<long>(1L), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<int>(1), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<short>(1), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<double>({{1.3}}) * static_cast<char>(1), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<long double>(1.4L), CRS<double>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<double>(1.4), CRS<double>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<float>(1.4F), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<long long>(1LL), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<long>(1L), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<int>(1), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<short>(1), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<float>({{1.3F}}) * static_cast<char>(1), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);

   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<long double>(1.4L), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<double>(1.4), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<float>(1.4F), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<long long>(2LL), CRS<long long>({{2LL}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<long>(2L), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<int>(2), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<short>(2), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long long>({{1LL}}) * static_cast<char>(2), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<long>({{1L}}) * static_cast<long double>(1.4L), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<double>(1.4), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<float>(1.4F), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<long long>(2LL), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<long>(2L), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<int>(2), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<short>(2), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<long>({{1L}}) * static_cast<char>(2), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<int>({{1}}) * static_cast<long double>(1.4L), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<double>(1.4), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<float>(1.4F), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<long long>(2LL), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<long>(2L), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<int>(2), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<short>(2), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<int>({{1}}) * static_cast<char>(2), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<short>({{1}}) * static_cast<long double>(1.4L), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<double>(1.4), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<float>(1.4F), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<long long>(2LL), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<long>(2L), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<int>(2), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<short>(2), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<short>({{1}}) * static_cast<char>(2), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(CRS<char>({{1}}) * static_cast<long double>(1.4L), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<double>(1.4), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<float>(1.4F), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<long long>(2LL), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<long>(2L), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<int>(2), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<short>(2), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(CRS<char>({{1}}) * static_cast<char>(2), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);
}

TEST(CRS, MultiplicationScalarMatrix) {
   auto expect_crs_eq = [](const auto &lhs, const auto &rhs, const auto &prec, const auto &line) {
      const bool c1 = std::is_integral<typename std::remove_cvref<decltype(lhs)>::type::ValueType>::value;
      const bool c2 = std::is_integral<typename std::remove_cvref<decltype(rhs)>::type::ValueType>::value;
      if (c1 && c2) {
         EXPECT_EQ(lhs, rhs);
      } else {
         EXPECT_EQ(lhs.row_dim, rhs.row_dim);
         EXPECT_EQ(lhs.col_dim, rhs.col_dim);
         EXPECT_EQ(lhs.row.size(), rhs.row.size());
         EXPECT_EQ(lhs.col.size(), rhs.col.size());
         EXPECT_EQ(lhs.val.size(), rhs.val.size());
         EXPECT_EQ(lhs.tag, rhs.tag);
         EXPECT_EQ(lhs.name, rhs.name);
         for (std::size_t i = 0; i < lhs.row.size(); ++i) {
            EXPECT_EQ(lhs.row.at(i), rhs.row.at(i));
         }
         for (std::size_t i = 0; i < lhs.col.size(); ++i) {
            EXPECT_EQ(lhs.col.at(i), rhs.col.at(i));
         }
         for (std::size_t i = 0; i < lhs.val.size(); ++i) {
            const auto lhs_val = std::abs(lhs.val.at(i) - rhs.val.at(i));
            const auto rhs_val = 4 * prec;
            EXPECT_TRUE(lhs_val <= rhs_val) << lhs << " !<= " << rhs << " from " << line;
         }
      }
   };
   using blas::CRS;
   expect_crs_eq(static_cast<long double>(1.4L) * CRS<long double>({{1.3L}}), CRS<long double>({{1.82L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<long double>({{1.3L}}), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<long double>({{1.3L}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(1LL) * CRS<long double>({{1.3L}}), CRS<long double>({{1.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<long>(1L) * CRS<long double>({{1.3L}}), CRS<long double>({{1.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<int>(1) * CRS<long double>({{1.3L}}), CRS<long double>({{1.3L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(1) * CRS<long double>({{1.3L}}), CRS<long double>({{1.3L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<char>(1) * CRS<long double>({{1.3L}}), CRS<long double>({{1.3L}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<double>({{1.3}}), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<double>({{1.3}}), CRS<double>({{1.82}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<double>({{1.3}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(1LL) * CRS<double>({{1.3}}), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long>(1L) * CRS<double>({{1.3}}), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(1) * CRS<double>({{1.3}}), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(1) * CRS<double>({{1.3}}), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(1) * CRS<double>({{1.3}}), CRS<double>({{1.3}}), DBL_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<float>({{1.3F}}), CRS<double>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<float>({{1.3F}}), CRS<double>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<float>({{1.3F}}), CRS<float>({{1.82F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(1LL) * CRS<float>({{1.3F}}), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long>(1L) * CRS<float>({{1.3F}}), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(1) * CRS<float>({{1.3F}}), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(1) * CRS<float>({{1.3F}}), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(1) * CRS<float>({{1.3F}}), CRS<double>({{1.3F}}), FLT_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<long long>({{1LL}}), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<long long>({{1LL}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<long long>({{1LL}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(2LL) * CRS<long long>({{1LL}}), CRS<long long>({{2LL}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<long>(2L) * CRS<long long>({{1LL}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(2) * CRS<long long>({{1LL}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(2) * CRS<long long>({{1LL}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(2) * CRS<long long>({{1LL}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<long>({{1L}}), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<long>({{1L}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<long>({{1L}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(2LL) * CRS<long>({{1L}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long>(2L) * CRS<long>({{1L}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(2) * CRS<long>({{1L}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(2) * CRS<long>({{1L}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(2) * CRS<long>({{1L}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<int>({{1}}), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<int>({{1}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<int>({{1}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(2LL) * CRS<int>({{1}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long>(2L) * CRS<int>({{1}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(2) * CRS<int>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(2) * CRS<int>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(2) * CRS<int>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<short>({{1}}), CRS<long double>({{1.4L}}), LDBL_EPSILON,
                 __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<short>({{1}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<short>({{1}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(2LL) * CRS<short>({{1}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long>(2L) * CRS<short>({{1}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(2) * CRS<short>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(2) * CRS<short>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(2) * CRS<short>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);

   expect_crs_eq(static_cast<long double>(1.4L) * CRS<char>({{1}}), CRS<long double>({{1.4L}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<double>(1.4) * CRS<char>({{1}}), CRS<double>({{1.4}}), DBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<float>(1.4F) * CRS<char>({{1}}), CRS<float>({{1.4F}}), FLT_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long long>(2LL) * CRS<char>({{1}}), CRS<long long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<long>(2L) * CRS<char>({{1}}), CRS<long>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<int>(2) * CRS<char>({{1}}), CRS<int>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<short>(2) * CRS<char>({{1}}), CRS<short>({{2LL}}), LDBL_EPSILON, __LINE__);
   expect_crs_eq(static_cast<char>(2) * CRS<char>({{1}}), CRS<char>({{2LL}}), LDBL_EPSILON, __LINE__);
}

TEST(CRS, MultiplicationEquality) {
   using blas::CRS;
   EXPECT_TRUE(CRS<long long>({{1LL}}) == CRS<long long>({{1LL}}));
   EXPECT_TRUE(CRS<long long>({{1LL}}) == CRS<long>({{1L}}));
   EXPECT_TRUE(CRS<long long>({{1LL}}) == CRS<int>({{1}}));
   EXPECT_TRUE(CRS<long long>({{1LL}}) == CRS<short>({{1}}));
   EXPECT_TRUE(CRS<long long>({{1LL}}) == CRS<char>({{1}}));

   EXPECT_TRUE(CRS<long>({{1L}}) == CRS<long long>({{1LL}}));
   EXPECT_TRUE(CRS<long>({{1L}}) == CRS<long>({{1L}}));
   EXPECT_TRUE(CRS<long>({{1L}}) == CRS<int>({{1}}));
   EXPECT_TRUE(CRS<long>({{1L}}) == CRS<short>({{1}}));
   EXPECT_TRUE(CRS<long>({{1L}}) == CRS<char>({{1}}));

   EXPECT_TRUE(CRS<int>({{1}}) == CRS<long long>({{1LL}}));
   EXPECT_TRUE(CRS<int>({{1}}) == CRS<long>({{1L}}));
   EXPECT_TRUE(CRS<int>({{1}}) == CRS<int>({{1}}));
   EXPECT_TRUE(CRS<int>({{1}}) == CRS<short>({{1}}));
   EXPECT_TRUE(CRS<int>({{1}}) == CRS<char>({{1}}));

   EXPECT_TRUE(CRS<short>({{1}}) == CRS<long long>({{1LL}}));
   EXPECT_TRUE(CRS<short>({{1}}) == CRS<long>({{1L}}));
   EXPECT_TRUE(CRS<short>({{1}}) == CRS<int>({{1}}));
   EXPECT_TRUE(CRS<short>({{1}}) == CRS<short>({{1}}));
   EXPECT_TRUE(CRS<short>({{1}}) == CRS<char>({{1}}));

   EXPECT_TRUE(CRS<char>({{1}}) == CRS<long long>({{1LL}}));
   EXPECT_TRUE(CRS<char>({{1}}) == CRS<long>({{1L}}));
   EXPECT_TRUE(CRS<char>({{1}}) == CRS<int>({{1}}));
   EXPECT_TRUE(CRS<char>({{1}}) == CRS<short>({{1}}));
   EXPECT_TRUE(CRS<char>({{1}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<long long>({{1LL, 2LL}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<long long>({{1LL, 2LL}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<long long>({{1LL, 2LL}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<long long>({{1LL, 2LL}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<long long>({{1LL, 2LL}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<long>({{1L, 2L}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<long>({{1L, 2L}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<long>({{1L, 2L}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<long>({{1L, 2L}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<long>({{1L, 2L}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<int>({{1, -1}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<int>({{1, -1}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<int>({{1, -1}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<int>({{1, -1}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<int>({{1, -1}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<short>({{1, -1}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<short>({{1, -1}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<short>({{1, -1}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<short>({{1, -1}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<short>({{1, -1}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<char>({{1, -1}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<char>({{1, -1}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<char>({{1, -1}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<char>({{1, -1}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<char>({{1, -1}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<long long>({{2LL}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<long long>({{2LL}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<long long>({{2LL}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<long long>({{2LL}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<long long>({{2LL}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<long>({{2L}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<long>({{2L}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<long>({{2L}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<long>({{2L}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<long>({{2L}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<int>({{2}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<int>({{2}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<int>({{2}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<int>({{2}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<int>({{2}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<short>({{2}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<short>({{2}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<short>({{2}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<short>({{2}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<short>({{2}}) == CRS<char>({{1}}));

   EXPECT_FALSE(CRS<char>({{2}}) == CRS<long long>({{1LL}}));
   EXPECT_FALSE(CRS<char>({{2}}) == CRS<long>({{1L}}));
   EXPECT_FALSE(CRS<char>({{2}}) == CRS<int>({{1}}));
   EXPECT_FALSE(CRS<char>({{2}}) == CRS<short>({{1}}));
   EXPECT_FALSE(CRS<char>({{2}}) == CRS<char>({{1}}));
}

} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_BLAS_CRS_HPP_ */
