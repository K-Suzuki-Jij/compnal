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
//  Created by Kohei Suzuki on 2022/06/10.
//

#ifndef COMPNAL_TEST_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_
#define COMPNAL_TEST_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_

#include "../../../src/model/classical/polynomial_ising.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(Model, PolynomialIsingBasicChain) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::PolynomialIsing<lattice::Chain, TestRealType>;
   
   auto lattice = lattice::Chain{4, BC::PBC};
   auto model = ModelType{lattice, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(model.GetSystemSize(), 4);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::PBC);
   EXPECT_EQ(model.GetInteraction().size(), 4);
   EXPECT_NEAR(model.GetInteraction().at(0), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(1), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(2), -3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(3), +4.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 3);
   
}

TEST(Model, PolynomialIsingBasicSquare) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::PolynomialIsing<lattice::Square, TestRealType>;
   
   auto lattice = lattice::Square{4, 3, BC::PBC};
   auto model = ModelType{lattice, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(model.GetSystemSize(), 12);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::PBC);
   EXPECT_EQ(model.GetInteraction().size(), 4);
   EXPECT_NEAR(model.GetInteraction().at(0), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(1), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(2), -3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(3), +4.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 3);
   
}

TEST(Model, PolynomialIsingBasicCubic) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::PolynomialIsing<lattice::Cubic, TestRealType>;
   
   auto lattice = lattice::Cubic{4, 3, 2, BC::PBC};
   auto model = ModelType{lattice, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(model.GetSystemSize(), 24);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::PBC);
   EXPECT_EQ(model.GetInteraction().size(), 4);
   EXPECT_NEAR(model.GetInteraction().at(0), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(1), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(2), -3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(3), +4.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 3);
   
}

TEST(Model, PolynomialIsingBasicInfiniteRange) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::PolynomialIsing<lattice::InfiniteRange, TestRealType>;
   
   auto lattice = lattice::InfiniteRange{4};
   auto model = ModelType{lattice, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(model.GetSystemSize(), 4);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::NONE);
   EXPECT_EQ(model.GetInteraction().size(), 4);
   EXPECT_NEAR(model.GetInteraction().at(0), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(1), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(2), -3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR(model.GetInteraction().at(3), +4.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetDegree(), 3);
   
}

TEST(Model, PolynomialIsingBasicAnyLattice) {
   using BC = lattice::BoundaryCondition;
   using ModelType = model::classical::PolynomialIsing<lattice::AnyLattice, TestRealType>;
   using Tup = utility::AnyTupleType;
   using InteractionType = interaction::classical::PolynomialAny<TestRealType>;
   
   const InteractionType::PolynomialType interaction = {
      {{0, 1}, -1.0},
      {{0, 1, 2}, +1.5},
      {{2, 3, 4, Tup{1, "a"}}, +2.0}
   };
   
   auto lattice = lattice::AnyLattice{};
   auto model = ModelType{lattice, interaction};
   
   EXPECT_EQ(model.GetSystemSize(), 6);
   EXPECT_EQ(model.GetBoundaryCondition(), BC::NONE);
   
   EXPECT_EQ(model.GetKeyValueList().size(), 3);
   EXPECT_EQ(model.GetKeyValueList().at(0).first.size(), 2);
   EXPECT_EQ(model.GetKeyValueList().at(0).first.at(0), 0);
   EXPECT_EQ(model.GetKeyValueList().at(0).first.at(1), 1);
   EXPECT_NEAR(model.GetKeyValueList().at(0).second, -1.0, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetKeyValueList().at(1).first.size(), 3);
   EXPECT_EQ(model.GetKeyValueList().at(1).first.at(0), 0);
   EXPECT_EQ(model.GetKeyValueList().at(1).first.at(1), 1);
   EXPECT_EQ(model.GetKeyValueList().at(1).first.at(2), 2);
   EXPECT_NEAR(model.GetKeyValueList().at(1).second, +1.5, test_epsilon<TestRealType>);
   EXPECT_EQ(model.GetKeyValueList().at(2).first.size(), 4);
   EXPECT_EQ(model.GetKeyValueList().at(2).first.at(0), 2);
   EXPECT_EQ(model.GetKeyValueList().at(2).first.at(1), 3);
   EXPECT_EQ(model.GetKeyValueList().at(2).first.at(2), 4);
   EXPECT_EQ(model.GetKeyValueList().at(2).first.at(3), 5);
   EXPECT_NEAR(model.GetKeyValueList().at(2).second, +2.0, test_epsilon<TestRealType>);
   
   EXPECT_EQ(model.GetAdjacencyList().size(), 6);
   EXPECT_EQ(model.GetAdjacencyList().at(0).size(), 2);
   EXPECT_EQ(model.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(model.GetAdjacencyList().at(0).at(1), 1);
   EXPECT_EQ(model.GetAdjacencyList().at(1).size(), 2);
   EXPECT_EQ(model.GetAdjacencyList().at(1).at(0), 0);
   EXPECT_EQ(model.GetAdjacencyList().at(1).at(1), 1);
   EXPECT_EQ(model.GetAdjacencyList().at(2).size(), 2);
   EXPECT_EQ(model.GetAdjacencyList().at(2).at(0), 1);
   EXPECT_EQ(model.GetAdjacencyList().at(2).at(1), 2);
   EXPECT_EQ(model.GetAdjacencyList().at(3).size(), 1);
   EXPECT_EQ(model.GetAdjacencyList().at(3).at(0), 2);
   EXPECT_EQ(model.GetAdjacencyList().at(4).size(), 1);
   EXPECT_EQ(model.GetAdjacencyList().at(4).at(0), 2);
   EXPECT_EQ(model.GetAdjacencyList().at(5).size(), 1);
   EXPECT_EQ(model.GetAdjacencyList().at(5).at(0), 2);
   
   EXPECT_EQ(model.GetIndexList(), (std::vector<InteractionType::IndexType>{0, 1, 2, 3, 4, Tup{1, "a"}}));
   
   EXPECT_EQ(model.GetIndexMap().size(), 6);
   EXPECT_EQ(model.GetIndexMap().at(0), 0);
   EXPECT_EQ(model.GetIndexMap().at(1), 1);
   EXPECT_EQ(model.GetIndexMap().at(2), 2);
   EXPECT_EQ(model.GetIndexMap().at(3), 3);
   EXPECT_EQ(model.GetIndexMap().at(4), 4);
   EXPECT_EQ(model.GetIndexMap().at(Tup{1, "a"}), 5);
  
   EXPECT_EQ(model.GetDegree(), 4);
}

TEST(Model, PolynomialIsingEnegyChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using Ising = model::classical::PolynomialIsing<Chain, TestRealType>;
   
   EXPECT_NEAR((Ising{Chain{3, BC::OBC}, {}        }.CalculateEnergy({-1,+1,-1})), +0.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::OBC}, {{0, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::OBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1})), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::OBC}, {{2, 2.0}}}.CalculateEnergy({-1,+1,-1})), -4.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::OBC}, {{3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::OBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +1.0, test_epsilon<TestRealType>);

   EXPECT_NEAR((Ising{Chain{3, BC::PBC}, {}        }.CalculateEnergy({-1,+1,-1})), +0.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::PBC}, {{0, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1})), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::PBC}, {{2, 2.0}}}.CalculateEnergy({-1,+1,-1})), -2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::PBC}, {{3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +6.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Chain{3, BC::PBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +5.0, test_epsilon<TestRealType>);
   
   EXPECT_THROW((Ising{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1}))      , std::runtime_error);
   EXPECT_THROW((Ising{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,+1,+1})), std::runtime_error);
   
}

TEST(Model, PolynomialIsingEnegySquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::PolynomialIsing<Square, TestRealType>;
   const std::vector<typename Ising::OPType> spins = {
      -1,+1,-1,+1,+1,+1
   };
   
   EXPECT_NEAR((Ising{Square{3, 2, BC::OBC}, {}}.CalculateEnergy(spins)), +0.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::OBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::OBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::OBC}, {{2, 2.0}}}.CalculateEnergy(spins)), -2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::OBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +4.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::OBC}, {{0, 3.0}, {1, 2.0}, {3, 2.0}}}.CalculateEnergy(spins)), +11.0, test_epsilon<TestRealType>);
   
   EXPECT_NEAR((Ising{Square{3, 2, BC::PBC}, {}}.CalculateEnergy(spins)), +0.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::PBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::PBC}, {{2, 2.0}}}.CalculateEnergy(spins)), +0.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::PBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +16.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Square{3, 2, BC::PBC}, {{0, 3.0}, {1, 2.0}, {3, 2.0}}}.CalculateEnergy(spins)), +23.0, test_epsilon<TestRealType>);
   
   EXPECT_THROW((Ising{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1}))      , std::runtime_error);
   EXPECT_THROW((Ising{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,+1,+1})), std::runtime_error);
   
}

TEST(Model, PolynomialIsingEnegyCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using Ising = model::classical::PolynomialIsing<Cubic, TestRealType>;
   const std::vector<typename Ising::OPType> spins = {
      -1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1
   };
   
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::OBC}, {}        }.CalculateEnergy(spins)), +0.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::OBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::OBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +4.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::OBC}, {{2, 2.0}}}.CalculateEnergy(spins)), +8.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::OBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +8.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::OBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy(spins)), +12.0, test_epsilon<TestRealType>);
   
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::PBC}, {}        }.CalculateEnergy(spins)), +0.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::PBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +4.0 , test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::PBC}, {{2, 2.0}}}.CalculateEnergy(spins)), +24.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::PBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +40.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy(spins)), +44.0, test_epsilon<TestRealType>);
   
   EXPECT_THROW((Ising{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1}))      , std::runtime_error);
   EXPECT_THROW((Ising{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,+1,+1})), std::runtime_error);
   
}

TEST(Model, PolynomialIsingEnegyInfiniteRange) {
   using Infinite = lattice::InfiniteRange;
   using Ising = model::classical::PolynomialIsing<Infinite, TestRealType>;

   EXPECT_NEAR((Ising{Infinite{3}, {{}      }}.CalculateEnergy({-1,+1,-1})), +0.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Infinite{3}, {{0, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Infinite{3}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1})), -1.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Infinite{3}, {{2, 2.0}}}.CalculateEnergy({-1,+1,-1})), -2.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Infinite{3}, {{3, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0, test_epsilon<TestRealType>);
   EXPECT_NEAR((Ising{Infinite{3}, {{1, 1.0}, {2, 2.0}, {3, 3.0}}}.CalculateEnergy({-1,+1,-1})), 0.0, test_epsilon<TestRealType>);
   
   EXPECT_THROW((Ising{Infinite{3}, {{3, 3.0}}}.CalculateEnergy({-1,+1}))      , std::runtime_error);
   EXPECT_THROW((Ising{Infinite{3}, {{3, 3.0}}}.CalculateEnergy({-1,+1,+1,+1})), std::runtime_error);
   
}

TEST(Model, PolynomialIsingEnegyAnyLattice) {
   using AnyLattice = lattice::AnyLattice;
   using Ising = model::classical::PolynomialIsing<AnyLattice, TestRealType>;
   using InteractionType = interaction::classical::PolynomialAny<TestRealType>;
   
   const InteractionType::PolynomialType interaction = {
      {{0, 1}, -1.0},
      {{0, 1, 2}, +1.5},
      {{2, 3, 4, 5}, +2.0}
   };
   
   EXPECT_NEAR((Ising{AnyLattice{}, interaction}.CalculateEnergy({-1,+1,-1,+1,+1,+1})), +0.5, test_epsilon<TestRealType>);
   EXPECT_THROW((Ising{AnyLattice{}, interaction}.CalculateEnergy({-1,+1})), std::runtime_error);
   EXPECT_THROW((Ising{AnyLattice{}, interaction}.CalculateEnergy({-1,+1,+1,+1,+1,+1,+1})), std::runtime_error);
   
}

} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_ */
