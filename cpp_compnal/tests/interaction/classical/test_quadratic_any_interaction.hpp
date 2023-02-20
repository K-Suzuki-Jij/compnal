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
//  test_quadratic_any_interaction.hpp
//  compnal
//
//  Created by kohei on 2022/10/18.
//  
//

#ifndef COMPNAL_TEST_INTERACTION_CLASSICAL_QUADRATIC_ANY_HPP_
#define COMPNAL_TEST_INTERACTION_CLASSICAL_QUADRATIC_ANY_HPP_

#include "../../../src/interaction/classical/quadratic_any.hpp"
#include "../../test_settings.hpp"

#include "gtest/gtest.h"

namespace compnal {
namespace test {

TEST(Interaction, ClassicalQuadraticAnyLinearOnly) {
   using InteractionType = interaction::classical::QuadraticAny<TestRealType>;
   
   const InteractionType::LinearType linear = {
      {1, 1.0},
      {2, 2.0},
      {3, 3.0}
   };
   
   const InteractionType interaction = InteractionType{linear, {}};
   
   EXPECT_EQ(interaction.GetSystemSize(), 3);
   EXPECT_EQ(interaction.GetDegree(), 1);
   EXPECT_EQ(interaction.GetConstant(), TestRealType{0.0});
   EXPECT_EQ(interaction.GetLinear(), (std::vector<TestRealType>{1.0, 2.0, 3.0}));
   
   EXPECT_EQ(interaction.GetRowPtr().size(), 4);
   EXPECT_EQ(interaction.GetColPtr().size(), 0);
   EXPECT_EQ(interaction.GetValPtr().size(), 0);
   
   EXPECT_EQ(interaction.GetRowPtr().at(0), 0);
   
   EXPECT_EQ(interaction.GetIndexList(), (std::vector<InteractionType::IndexType>{1, 2, 3}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 3);
   EXPECT_EQ(interaction.GetIndexMap().at(1), 0);
   EXPECT_EQ(interaction.GetIndexMap().at(2), 1);
   EXPECT_EQ(interaction.GetIndexMap().at(3), 2);
}

TEST(Interaction, ClassicalQuadraticAnyQuadOnly) {
   using InteractionType = interaction::classical::QuadraticAny<TestRealType>;
   
   const InteractionType::QuadraticType quadratic = {
      {{1, 1}, +3.0},
      {{1, 2}, -1.0},
      {{2, 1}, -1.5},
      {{3, 2}, +1.0}
   };
   
   const InteractionType interaction = InteractionType{{}, quadratic};
   
   EXPECT_EQ(interaction.GetSystemSize(), 3);
   EXPECT_EQ(interaction.GetDegree(), 2);
   EXPECT_EQ(interaction.GetConstant(), TestRealType{3.0});
   EXPECT_EQ(interaction.GetLinear(), (std::vector<TestRealType>{0.0, 0.0, 0.0}));
   
   EXPECT_EQ(interaction.GetRowPtr().size(), 4);
   EXPECT_EQ(interaction.GetColPtr().size(), 4);
   EXPECT_EQ(interaction.GetValPtr().size(), 4);
   
   EXPECT_EQ(interaction.GetRowPtr().at(0), 0);
   EXPECT_EQ(interaction.GetRowPtr().at(1), 1);
   EXPECT_EQ(interaction.GetRowPtr().at(2), 3);
   EXPECT_EQ(interaction.GetRowPtr().at(3), 4);
   
   EXPECT_EQ(interaction.GetColPtr().at(0), 1);
   EXPECT_EQ(interaction.GetColPtr().at(1), 0);
   EXPECT_EQ(interaction.GetColPtr().at(2), 2);
   EXPECT_EQ(interaction.GetColPtr().at(3), 1);
   
   EXPECT_NEAR(interaction.GetValPtr().at(0), TestRealType{-2.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(1), TestRealType{-2.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(2), TestRealType{+1.0}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(3), TestRealType{+1.0}, test_epsilon<TestRealType>);
   
   EXPECT_EQ(interaction.GetIndexList(), (std::vector<InteractionType::IndexType>{1, 2, 3}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 3);
   EXPECT_EQ(interaction.GetIndexMap().at(1), 0);
   EXPECT_EQ(interaction.GetIndexMap().at(2), 1);
   EXPECT_EQ(interaction.GetIndexMap().at(3), 2);
   
}

TEST(Interaction, ClassicalQuadraticAnyMix) {
   using Tup = utility::AnyTupleType;
   using InteractionType = interaction::classical::QuadraticAny<TestRealType>;
   
   const InteractionType::LinearType linear = {
      {1, 1.0},
      {"a", 2.0},
      {Tup{2, "b"}, 3}
   };
   
   const InteractionType::QuadraticType quadratic = {
      {{1, 1}, +3.0},
      {{1, 2}, -1.0},
      {{"a", 1}, -1.5},
      {{Tup{2, "b"}, Tup{2, "a"}}, -2.5}
   };
   
   // Sorted index list is {1, 2, a, 2a, 2b}.
   const InteractionType interaction = InteractionType{linear, quadratic};

   EXPECT_EQ(interaction.GetSystemSize(), 5);
   EXPECT_EQ(interaction.GetDegree(), 2);
   EXPECT_EQ(interaction.GetConstant(), TestRealType{3.0});
   EXPECT_EQ(interaction.GetLinear(), (std::vector<TestRealType>{1.0, 0.0, 2.0, 0.0, 3}));
   
   EXPECT_EQ(interaction.GetRowPtr().size(), 6);
   EXPECT_EQ(interaction.GetColPtr().size(), 6);
   EXPECT_EQ(interaction.GetValPtr().size(), 6);

   EXPECT_EQ(interaction.GetRowPtr().at(0), 0);
   EXPECT_EQ(interaction.GetRowPtr().at(1), 2);
   EXPECT_EQ(interaction.GetRowPtr().at(2), 3);
   EXPECT_EQ(interaction.GetRowPtr().at(3), 4);
   EXPECT_EQ(interaction.GetRowPtr().at(4), 5);
   EXPECT_EQ(interaction.GetRowPtr().at(5), 6);
   
   EXPECT_EQ(interaction.GetColPtr().at(0), 1);
   EXPECT_EQ(interaction.GetColPtr().at(1), 2);
   EXPECT_EQ(interaction.GetColPtr().at(2), 0);
   EXPECT_EQ(interaction.GetColPtr().at(3), 0);
   EXPECT_EQ(interaction.GetColPtr().at(4), 4);
   EXPECT_EQ(interaction.GetColPtr().at(5), 3);
   
   EXPECT_NEAR(interaction.GetValPtr().at(0), TestRealType{-1.0}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(1), TestRealType{-1.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(2), TestRealType{-1.0}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(3), TestRealType{-1.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(4), TestRealType{-2.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(interaction.GetValPtr().at(5), TestRealType{-2.5}, test_epsilon<TestRealType>);
   
   EXPECT_EQ(interaction.GetIndexList(),
             (std::vector<InteractionType::IndexType>{1, 2, "a", Tup{2, "a"}, Tup{2, "b"}}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 5);
   EXPECT_EQ(interaction.GetIndexMap().at(1)          , 0);
   EXPECT_EQ(interaction.GetIndexMap().at(2)          , 1);
   EXPECT_EQ(interaction.GetIndexMap().at("a")        , 2);
   EXPECT_EQ(interaction.GetIndexMap().at(Tup{2, "a"}), 3);
   EXPECT_EQ(interaction.GetIndexMap().at(Tup{2, "b"}), 4);
   
}

TEST(Interaction, ClassicalQuadraticAnyEmpty) {
   using InteractionType = interaction::classical::QuadraticAny<TestRealType>;
   const InteractionType interaction = InteractionType{{}, {}};
   
   EXPECT_EQ(interaction.GetSystemSize(), 0);
   EXPECT_EQ(interaction.GetDegree(), 0);
   EXPECT_EQ(interaction.GetConstant(), TestRealType{0.0});
   EXPECT_EQ(interaction.GetLinear(), (std::vector<TestRealType>{}));
   
   EXPECT_EQ(interaction.GetRowPtr().size(), 1);
   EXPECT_EQ(interaction.GetColPtr().size(), 0);
   EXPECT_EQ(interaction.GetValPtr().size(), 0);
   
   EXPECT_EQ(interaction.GetRowPtr().at(0), 0);
   
   EXPECT_EQ(interaction.GetIndexList(), (std::vector<InteractionType::IndexType>{}));
   
   EXPECT_EQ(interaction.GetIndexMap().size(), 0);

}

} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_INTERACTION_CLASSICAL_QUADRATIC_ANY_HPP_ */
