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
//  test_polynomial_any_interaction.hpp
//  compnal
//
//  Created by kohei on 2022/10/18.
//  
//

#ifndef COMPNAL_TEST_INTERACTION_POLYNOMIAL_ANY_HPP_
#define COMPNAL_TEST_INTERACTION_POLYNOMIAL_ANY_HPP_

#include "../../src/interaction/polynomial_any_interaction.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(InteractionPolynomialAny, Basic) {
   using PolyClass = interaction::PolynomialAnyInteraction<double>;
   
   const PolyClass::PolynomialType poly = {
      {{0, 1, 2}, -1.0},
      {{"a", 0, "b"}, +1.5},
      {{0, utility::AnyTupleType{2, "b"}, utility::AnyTupleType{2, "a"}}, +2.0}
   };
   
   const auto poly_interaction = PolyClass{poly};
   auto index_list = std::vector<PolyClass::IndexType>{
      0, 1, 2, "a", "b",
      utility::AnyTupleType{2, "b"}, utility::AnyTupleType{2, "a"}
   };
   
   std::sort(index_list.begin(), index_list.end());
   
   EXPECT_EQ(poly_interaction.GetSystemSize(), 7);
   EXPECT_EQ(poly_interaction.GetDegree(), 3);
   EXPECT_EQ(poly_interaction.GetIndexList(), index_list);
   for (const auto &it: poly_interaction.GetIndexMap()) {
      EXPECT_EQ(it.first, index_list[it.second]);
   }
}

TEST(InteractionPolynomialAny, IntKeys) {
   using PolyClass = interaction::PolynomialAnyInteraction<double>;
   
   const PolyClass::PolynomialType poly = {
      {{0, 1}, -1.0},
      {{0, 1, 2}, +1.5},
      {{2, 3, 4, 5}, +2.0}
   };
   
   const auto poly_interaction = PolyClass{poly};
   EXPECT_EQ(poly_interaction.GetSystemSize(), 6);
   EXPECT_EQ(poly_interaction.GetDegree(), 4);
   EXPECT_EQ(poly_interaction.GetIndexList(), (std::vector<PolyClass::IndexType>{0, 1, 2, 3, 4, 5}));
   EXPECT_EQ(poly_interaction.GetKeyList().size(), 3);
   EXPECT_EQ(poly_interaction.GetValueList().size(), 3);
   EXPECT_EQ(poly_interaction.GetKeyList().at(0).size(), 2);
   EXPECT_EQ(poly_interaction.GetKeyList().at(0).at(0), 0);
   EXPECT_EQ(poly_interaction.GetKeyList().at(0).at(1), 1);
   EXPECT_DOUBLE_EQ(poly_interaction.GetValueList().at(0), -1.0);
   EXPECT_EQ(poly_interaction.GetKeyList().at(1).size(), 3);
   EXPECT_EQ(poly_interaction.GetKeyList().at(1).at(0), 0);
   EXPECT_EQ(poly_interaction.GetKeyList().at(1).at(1), 1);
   EXPECT_EQ(poly_interaction.GetKeyList().at(1).at(2), 2);
   EXPECT_DOUBLE_EQ(poly_interaction.GetValueList().at(1), +1.5);
   EXPECT_EQ(poly_interaction.GetKeyList().at(2).size(), 4);
   EXPECT_EQ(poly_interaction.GetKeyList().at(2).at(0), 2);
   EXPECT_EQ(poly_interaction.GetKeyList().at(2).at(1), 3);
   EXPECT_EQ(poly_interaction.GetKeyList().at(2).at(2), 4);
   EXPECT_EQ(poly_interaction.GetKeyList().at(2).at(3), 5);
   EXPECT_DOUBLE_EQ(poly_interaction.GetValueList().at(2), +2.0);

   EXPECT_EQ(poly_interaction.GetAdjacencyList().size(), 6);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(0).size(), 2);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(0).at(1), 1);
   
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(1).size(), 2);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(1).at(0), 0);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(1).at(1), 1);
   
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(2).size(), 2);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(2).at(0), 1);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(2).at(1), 2);
   
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(3).size(), 1);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(3).at(0), 2);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(4).size(), 1);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(4).at(0), 2);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(5).size(), 1);
   EXPECT_EQ(poly_interaction.GetAdjacencyList().at(5).at(0), 2);

}



} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_INTERACTION_POLYNOMIAL_ANY_HPP_ */
