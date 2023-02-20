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
//  test_polynomial_any_interaction.hpp
//  compnal
//
//  Created by kohei on 2022/10/18.
//  
//

#ifndef COMPNAL_TEST_INTERACTION_CLASSICAL_POLYNOMIAL_ANY_HPP_
#define COMPNAL_TEST_INTERACTION_CLASSICAL_POLYNOMIAL_ANY_HPP_

#include "../../../src/interaction/classical/polynomial_any.hpp"
#include "../../test_settings.hpp"

namespace compnal {
namespace test {

TEST(Interaction, ClassicalPolynomialAnyLinear) {
   using InteractionType = interaction::classical::PolynomialAny<TestRealType>;
   
   const InteractionType::PolynomialType poly = {
      {{"a"}, -1.0},
      {{"b"}, +1.5},
      {{"c"}, +2.0}
   };
   
   const InteractionType interaction = InteractionType{poly};
   
   EXPECT_EQ(interaction.GetSystemSize(), 3);
   EXPECT_EQ(interaction.GetDegree(), 1);
   
   EXPECT_EQ(interaction.GetKeyValueList().size(), 3);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.size(), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.size(), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.size(), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.at(0), 0);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.at(0), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.at(0), 2);
   EXPECT_NEAR(interaction.GetKeyValueList().at(0).second, -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetKeyValueList().at(1).second, +1.5, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetKeyValueList().at(2).second, +2.0, test_epsilon<TestRealType>);
   
   EXPECT_EQ(interaction.GetIndexList(), (std::vector<InteractionType::IndexType>{"a", "b", "c"}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 3);
   EXPECT_EQ(interaction.GetIndexMap().at("a"), 0);
   EXPECT_EQ(interaction.GetIndexMap().at("b"), 1);
   EXPECT_EQ(interaction.GetIndexMap().at("c"), 2);
   
   EXPECT_EQ(interaction.GetAdjacencyList().size(), 3);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).at(0), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).at(0), 2);
   
}

TEST(Interaction, ClassicalPolynomialAnyQuadratic) {
   using Tup = utility::AnyTupleType;
   using InteractionType = interaction::classical::PolynomialAny<TestRealType>;
   
   const InteractionType::PolynomialType poly = {
      {{"a", "b"}, -1.0},
      {{"b", "a"}, +1.5},
      {{"b", Tup{1, "a"}}, +2.0}
   };
   
   const InteractionType interaction = InteractionType{poly};
   
   EXPECT_EQ(interaction.GetSystemSize(), 3);
   EXPECT_EQ(interaction.GetDegree(), 2);
   
   EXPECT_EQ(interaction.GetKeyValueList().size(), 2);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.size(), 2);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.size(), 2);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.at(0), 0);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.at(1), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.at(0), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.at(1), 2);
   EXPECT_NEAR(interaction.GetKeyValueList().at(0).second, 0.5, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetKeyValueList().at(1).second, 2.0, test_epsilon<TestRealType>);
   
   EXPECT_EQ(interaction.GetIndexList(), (std::vector<InteractionType::IndexType>{"a", "b", Tup{1, "a"}}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 3);
   EXPECT_EQ(interaction.GetIndexMap().at("a"), 0);
   EXPECT_EQ(interaction.GetIndexMap().at("b"), 1);
   EXPECT_EQ(interaction.GetIndexMap().at(Tup{1, "a"}), 2);
   
   EXPECT_EQ(interaction.GetAdjacencyList().size(), 3);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).size(), 2);
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).at(0), 0);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).at(1), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).at(0), 1);
   
}

TEST(Interaction, ClassicalPolynomialAnyPoly) {
   using Tup = utility::AnyTupleType;
   using InteractionType = interaction::classical::PolynomialAny<TestRealType>;
   
   const InteractionType::PolynomialType poly = {
      {{0, 1}, -1.0},
      {{0, 1, 2}, +1.5},
      {{2, 3, 4, Tup{1, "a"}}, +2.0}
   };
   
   const InteractionType interaction = InteractionType{poly};
   EXPECT_EQ(interaction.GetSystemSize(), 6);
   EXPECT_EQ(interaction.GetDegree(), 4);
   
   EXPECT_EQ(interaction.GetKeyValueList().size(), 3);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.size(), 2);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.at(0), 0);
   EXPECT_EQ(interaction.GetKeyValueList().at(0).first.at(1), 1);
   EXPECT_NEAR(interaction.GetKeyValueList().at(0).second, -1.0, test_epsilon<TestRealType>);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.size(), 3);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.at(0), 0);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.at(1), 1);
   EXPECT_EQ(interaction.GetKeyValueList().at(1).first.at(2), 2);
   EXPECT_NEAR(interaction.GetKeyValueList().at(1).second, +1.5, test_epsilon<TestRealType>);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.size(), 4);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.at(0), 2);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.at(1), 3);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.at(2), 4);
   EXPECT_EQ(interaction.GetKeyValueList().at(2).first.at(3), 5);
   EXPECT_NEAR(interaction.GetKeyValueList().at(2).second, +2.0, test_epsilon<TestRealType>);
   
   EXPECT_EQ(interaction.GetIndexList(), (std::vector<InteractionType::IndexType>{0, 1, 2, 3, 4, Tup{1, "a"}}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 6);
   EXPECT_EQ(interaction.GetIndexMap().at(0), 0);
   EXPECT_EQ(interaction.GetIndexMap().at(1), 1);
   EXPECT_EQ(interaction.GetIndexMap().at(2), 2);
   EXPECT_EQ(interaction.GetIndexMap().at(3), 3);
   EXPECT_EQ(interaction.GetIndexMap().at(4), 4);
   EXPECT_EQ(interaction.GetIndexMap().at(Tup{1, "a"}), 5);

   EXPECT_EQ(interaction.GetAdjacencyList().size(), 6);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).size(), 2);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(interaction.GetAdjacencyList().at(0).at(1), 1);
   
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).size(), 2);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).at(0), 0);
   EXPECT_EQ(interaction.GetAdjacencyList().at(1).at(1), 1);
   
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).size(), 2);
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).at(0), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(2).at(1), 2);
   
   EXPECT_EQ(interaction.GetAdjacencyList().at(3).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(3).at(0), 2);
   EXPECT_EQ(interaction.GetAdjacencyList().at(4).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(4).at(0), 2);
   EXPECT_EQ(interaction.GetAdjacencyList().at(5).size(), 1);
   EXPECT_EQ(interaction.GetAdjacencyList().at(5).at(0), 2);

}


} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_INTERACTION_CLASSICAL_POLYNOMIAL_ANY_HPP_ */
