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
//  test_ising.hpp
//  compnal
//
//  Created by kohei on 2022/08/28.
//  
//

#ifndef COMPNAL_TEST_MODEL_ISING_HPP_
#define COMPNAL_TEST_MODEL_ISING_HPP_

#include "../../../src/model/classical/ising.hpp"

namespace compnal {
namespace test {

TEST(ModelIsing, AnyLatticeBasic) {
   using IsingClass = model::Ising<lattice::AnyLattice, double>;
   using LinearType = IsingClass::LinearType;
   using QuadType = IsingClass::QuadraticType;
   
   const LinearType linear = {
      {1, 1.0},
      {"a", 1.0},
      {utility::AnyTupleType{2, "b"}, 1.5}
   };
   
   const QuadType quadratic = {
      {{1, 1}, +3.0},
      {{1, 2}, -1.0},
      {{"a", 1}, -1.5},
      {{utility::AnyTupleType{2, "b"}, utility::AnyTupleType{2, "a"}}, -1.5}
   };
   
   const IsingClass ising{lattice::AnyLattice{}, linear, quadratic};
   auto index_list = std::vector<IsingClass::IndexType>{1, 2, "a", utility::AnyTupleType{2, "b"}, utility::AnyTupleType{2, "a"}};
   std::sort(index_list.begin(), index_list.end());
   
   EXPECT_EQ(ising.GetSystemSize(), 5);
   EXPECT_EQ(ising.GetDegree(), 2);
   EXPECT_EQ(ising.GenerateIndexList(), index_list);
   for (const auto &it: ising.GetIndexMap()) {
      EXPECT_EQ(it.first, index_list[it.second]);
   }
   EXPECT_EQ(ising.GetConstant(), 3.0);
   for (const auto &it: linear) {
      EXPECT_DOUBLE_EQ(ising.GetLinear().at(ising.GetIndexMap().at(it.first)), it.second);
   }
   EXPECT_EQ(ising.GetBoundaryCondition(), lattice::BoundaryCondition::NONE);
   EXPECT_THROW((ising.CalculateEnergy({})), std::runtime_error);
   EXPECT_THROW((ising.CalculateMoment({{}}, 0)), std::runtime_error);
   EXPECT_THROW((ising.CalculateOnsiteAverage({{}}, -1)), std::out_of_range);
}

TEST(ModelIsing, AnyLatticeInt) {
   using IsingClass = model::Ising<lattice::AnyLattice, double>;
   using LinearType = IsingClass::LinearType;
   using QuadType = IsingClass::QuadraticType;
   
   const LinearType linear = {
      {1, 1.0},
      {2, 1.0},
      {3, 1.5}
   };
   
   const QuadType quadratic = {
      {{1, 1}, +3.0},
      {{1, 2}, -1.0},
      {{2, 1}, -1.5},
      {{3, 2}, +1.0}
   };
   
   const IsingClass ising{lattice::AnyLattice{}, linear, quadratic};
   EXPECT_DOUBLE_EQ((ising.CalculateEnergy({-1, +1, -1})), 3.0);
   
   EXPECT_EQ(ising.GetRowPtr().size(), 4);
   EXPECT_EQ(ising.GetRowPtr().at(0), 0);
   EXPECT_EQ(ising.GetRowPtr().at(1), 1);
   EXPECT_EQ(ising.GetRowPtr().at(2), 3);
   EXPECT_EQ(ising.GetRowPtr().at(3), 4);
   
   EXPECT_EQ(ising.GetColPtr().size(), 4);
   EXPECT_EQ(ising.GetColPtr().at(0), 1);
   EXPECT_EQ(ising.GetColPtr().at(1), 0);
   EXPECT_EQ(ising.GetColPtr().at(2), 2);
   EXPECT_EQ(ising.GetColPtr().at(3), 1);
   
   EXPECT_DOUBLE_EQ(ising.GetValPtr().size(), 4);
   EXPECT_DOUBLE_EQ(ising.GetValPtr().at(0), -2.5);
   EXPECT_DOUBLE_EQ(ising.GetValPtr().at(1), -2.5);
   EXPECT_DOUBLE_EQ(ising.GetValPtr().at(2), 1);
   EXPECT_DOUBLE_EQ(ising.GetValPtr().at(3), 1);
   
   EXPECT_DOUBLE_EQ((ising.CalculateMoment({{-1, +1, -1}, {+1, +1, +1}}, 1)), 1);
   EXPECT_DOUBLE_EQ((ising.CalculateMoment({{-1, +1, -1}, {+1, +1, +1}}, 2)), 5);
   
   EXPECT_DOUBLE_EQ((ising.CalculateOnsiteAverage({{-1, +1, -1}, {+1, +1, +1}}, 1)), 0);
   EXPECT_DOUBLE_EQ((ising.CalculateOnsiteAverage({{-1, +1, -1}, {+1, +1, +1}}, 2)), 1);
   EXPECT_DOUBLE_EQ((ising.CalculateOnsiteAverage({{-1, +1, -1}, {+1, +1, +1}}, 3)), 0);
   
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 1, 1)), 1);
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 1, 2)), 0);
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 1, 3)), 1);
   
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 2, 1)), 0);
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 2, 2)), 1);
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 2, 3)), 0);
   
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 3, 1)), 1);
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 3, 2)), 0);
   EXPECT_DOUBLE_EQ((ising.CalculateCorrelation({{-1, +1, -1}, {+1, +1, +1}}, 3, 3)), 1);
}

TEST(ModelIsing, Square) {
   
   const auto e_pbc = model::make_ising<lattice::Square, double>(lattice::Square{2, 3, lattice::BoundaryCondition::PBC}, 0, -2).CalculateEnergy({1, -1, 1, 1, -1, 1});
   const auto e_obc = model::make_ising<lattice::Square, double>(lattice::Square{2, 3, lattice::BoundaryCondition::OBC}, 0, -2).CalculateEnergy({1, -1, 1, 1, -1, 1});
   EXPECT_DOUBLE_EQ(e_obc, 2.0);
   EXPECT_DOUBLE_EQ(e_pbc, 8.0);

   
}


} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_MODEL_POLYNOMIAL_ISING_HPP_ */
