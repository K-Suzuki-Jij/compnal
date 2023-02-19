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

TEST(ModelClassicalIsing, BasicChain) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::Ising<lattice::Chain, TestRealType>;
   
   auto lattice = lattice::Chain{4, BC::PBC};
   auto model = ModelType{lattice, 1.0, 2.0};
   
   EXPECT_EQ(model.GetSystemSize(), 4);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::PBC);
   EXPECT_NEAR(model.GetLinear(), 1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetQuadratic(), 2.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 2);
      
}

TEST(ModelClassicalIsing, BasicSquare) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::Ising<lattice::Square, TestRealType>;
   
   auto lattice = lattice::Square{4, 2, BC::PBC};
   auto model = ModelType{lattice, 1.0, 2.0};
   
   EXPECT_EQ(model.GetSystemSize(), 8);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::PBC);
   EXPECT_NEAR(model.GetLinear(), 1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetQuadratic(), 2.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 2);
      
}

TEST(ModelClassicalIsing, BasicCubic) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::Ising<lattice::Cubic, TestRealType>;
   
   auto lattice = lattice::Cubic{4, 2, 3, BC::PBC};
   auto model = ModelType{lattice, 1.0, 2.0};
   
   EXPECT_EQ(model.GetSystemSize(), 24);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::PBC);
   EXPECT_NEAR(model.GetLinear(), 1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetQuadratic(), 2.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 2);
      
}

TEST(ModelClassicalIsing, BasicInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::Ising<lattice::InfiniteRange, TestRealType>;
   
   auto lattice = lattice::InfiniteRange{4};
   auto model = ModelType{lattice, 1.0, 2.0};
   
   EXPECT_EQ(model.GetSystemSize(), 4);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::NONE);
   EXPECT_NEAR(model.GetLinear(), 1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetQuadratic(), 2.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 2);
      
}

TEST(ModelClassicalIsing, BasicAnyLattice) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::Ising<lattice::AnyLattice, TestRealType>;
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
   
   auto lattice = lattice::AnyLattice{};
   
   auto model = ModelType{lattice, linear, quadratic};
   
   EXPECT_EQ(model.GetSystemSize(), 5);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::NONE);
   EXPECT_NEAR(model.GetConstant(), TestRealType{3.0}, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 2);
   EXPECT_EQ(model.GetLinear(), (std::vector<TestRealType>{1.0, 0.0, 2.0, 0.0, 3.0}));
   
   EXPECT_EQ(model.GetRowPtr().size(), 6);
   EXPECT_EQ(model.GetColPtr().size(), 6);
   EXPECT_EQ(model.GetValPtr().size(), 6);

   EXPECT_EQ(model.GetRowPtr().at(0), 0);
   EXPECT_EQ(model.GetRowPtr().at(1), 2);
   EXPECT_EQ(model.GetRowPtr().at(2), 3);
   EXPECT_EQ(model.GetRowPtr().at(3), 4);
   EXPECT_EQ(model.GetRowPtr().at(4), 5);
   EXPECT_EQ(model.GetRowPtr().at(5), 6);
   
   EXPECT_EQ(model.GetColPtr().at(0), 1);
   EXPECT_EQ(model.GetColPtr().at(1), 2);
   EXPECT_EQ(model.GetColPtr().at(2), 0);
   EXPECT_EQ(model.GetColPtr().at(3), 0);
   EXPECT_EQ(model.GetColPtr().at(4), 4);
   EXPECT_EQ(model.GetColPtr().at(5), 3);
   
   EXPECT_NEAR(model.GetValPtr().at(0), TestRealType{-1.0}, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetValPtr().at(1), TestRealType{-1.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetValPtr().at(2), TestRealType{-1.0}, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetValPtr().at(3), TestRealType{-1.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetValPtr().at(4), TestRealType{-2.5}, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetValPtr().at(5), TestRealType{-2.5}, test_epsilon<TestRealType>);
   
   EXPECT_EQ(model.GetIndexList(),
             (std::vector<InteractionType::IndexType>{1, 2, "a", Tup{2, "a"}, Tup{2, "b"}}));
   
   EXPECT_EQ(model.GetIndexMap().size(), 5);
   EXPECT_EQ(model.GetIndexMap().at(1)          , 0);
   EXPECT_EQ(model.GetIndexMap().at(2)          , 1);
   EXPECT_EQ(model.GetIndexMap().at("a")        , 2);
   EXPECT_EQ(model.GetIndexMap().at(Tup{2, "a"}), 3);
   EXPECT_EQ(model.GetIndexMap().at(Tup{2, "b"}), 4);
}

TEST(ModelClassicalIsing, Enegy) {
   using BC = lattice::BoundaryCondition;
   
   using Chain = lattice::Chain;
   using Square = lattice::Square;
   using Cubic = lattice::Cubic;
   using Infinite = lattice::InfiniteRange;
   using Any = lattice::AnyLattice;
   
   using IsingChain = model::classical::Ising<Chain, TestRealType>;
   
   //EXPECT_NEAR((IsingChain{Chain{3, BC::OBC}, 1.0, 2.0}), 3.0, test_epsilon<TestRealType>);
   
   
}

} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_MODEL_POLYNOMIAL_ISING_HPP_ */
