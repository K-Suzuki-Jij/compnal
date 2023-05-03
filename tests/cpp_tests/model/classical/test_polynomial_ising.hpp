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
//  test_polynomial_ising.hpp
//  compnal
//
//  Created by kohei on 2023/05/03.
//  
//

#ifndef COMPNAL_TEST_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_
#define COMPNAL_TEST_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_

#include "../../../../include/lattice/all.hpp"
#include "../../../../include/model/classical/polynomial_ising.hpp"

namespace compnal {
namespace test {

TEST(ModelClassical, PolynomialIsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using PIsing = model::classical::PolynomialIsing<Chain>;
   
   Chain chain{4, BC::OBC};
   PIsing pising{chain, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   EXPECT_DOUBLE_EQ(pising.GetInteraction().size(), 4);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(0), -1.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(1), +2.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(2), -3.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(3), +4.0);
   EXPECT_EQ(pising.GetDegree(), 3);
   EXPECT_NO_THROW(pising.GetLattice());
   
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {}        }.CalculateEnergy({-1,+1,-1})), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{0, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1})), -1.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{2, 2.0}}}.CalculateEnergy({-1,+1,-1})), -4.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +2.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +1.0);
   
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {}        }.CalculateEnergy({-1,+1,-1})), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{0, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1})), -1.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{2, 2.0}}}.CalculateEnergy({-1,+1,-1})), -2.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +6.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy({-1,+1,-1})), +5.0);
   
   EXPECT_THROW((PIsing{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,+1,+1})), std::range_error);
}

TEST(ModelClassical, PolynomialIsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using PIsing = model::classical::PolynomialIsing<Square>;
   const std::vector<typename PIsing::PHQType> spins = {
      -1,+1,-1,+1,+1,+1
   };
   
   Square square{4, 3, BC::PBC};
   PIsing pising{square, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(pising.GetInteraction().size(), 4);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(0), -1.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(1), +2.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(2), -3.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(3), +4.0);
   EXPECT_EQ(pising.GetDegree(), 3);
   EXPECT_NO_THROW(pising.GetLattice());
   
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::OBC}, {}}.CalculateEnergy(spins)), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::OBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::OBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +2.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::OBC}, {{2, 2.0}}}.CalculateEnergy(spins)), -2.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::OBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +4.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::OBC}, {{0, 3.0}, {1, 2.0}, {3, 2.0}}}.CalculateEnergy(spins)), +11.0);
   
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::PBC}, {}}.CalculateEnergy(spins)), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::PBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +2.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::PBC}, {{2, 2.0}}}.CalculateEnergy(spins)), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::PBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +16.0);
   EXPECT_DOUBLE_EQ((PIsing{Square{3, 2, BC::PBC}, {{0, 3.0}, {1, 2.0}, {3, 2.0}}}.CalculateEnergy(spins)), +23.0);
   
   EXPECT_THROW((PIsing{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,+1,+1,+1,+1,+1})), std::range_error);
}

TEST(ModelClassical, PolynomialIsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using PIsing = model::classical::PolynomialIsing<Cubic>;
   const std::vector<typename PIsing::PHQType> spins = {
      -1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1
   };
   
   Cubic cubic{4, 3, 2, BC::PBC};
   PIsing pising{cubic, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(pising.GetInteraction().size(), 4);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(0), -1.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(1), +2.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(2), -3.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(3), +4.0);
   EXPECT_EQ(pising.GetDegree(), 3);
   EXPECT_NO_THROW(pising.GetLattice());
   
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::OBC}, {}        }.CalculateEnergy(spins)), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::OBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::OBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +4.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::OBC}, {{2, 2.0}}}.CalculateEnergy(spins)), +8.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::OBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +8.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::OBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy(spins)), +12.0);
   
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::PBC}, {}        }.CalculateEnergy(spins)), +0.0 );
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::PBC}, {{0, 3.0}}}.CalculateEnergy(spins)), +3.0 );
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(spins)), +4.0 );
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::PBC}, {{2, 2.0}}}.CalculateEnergy(spins)), +24.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::PBC}, {{3, 2.0}}}.CalculateEnergy(spins)), +40.0);
   EXPECT_DOUBLE_EQ((PIsing{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy(spins)), +44.0);
   
   EXPECT_THROW((PIsing{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1,+1})), std::range_error);
}

TEST(ModelClassical, PolynomialIsingOnInfiniteRang) {
   using Infinite = lattice::InfiniteRange;
   using PIsing = model::classical::PolynomialIsing<Infinite>;
   
   Infinite infinite{4};
   PIsing pising{infinite, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}};
   
   EXPECT_EQ(pising.GetInteraction().size(), 4);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(0), -1.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(1), +2.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(2), -3.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(3), +4.0);
   EXPECT_EQ(pising.GetDegree(), 3);
   
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{}      }}.CalculateEnergy({-1,+1,-1})), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{0, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{1, 1.0}}}.CalculateEnergy({-1,+1,-1})), -1.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{2, 2.0}}}.CalculateEnergy({-1,+1,-1})), -2.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{3, 3.0}}}.CalculateEnergy({-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{1, 1.0}, {2, 2.0}, {3, 3.0}}}.CalculateEnergy({-1,+1,-1})), 0.0);
   
   EXPECT_THROW((PIsing{Infinite{3}, {{3, 3.0}}}.CalculateEnergy({-1,+1}))      , std::runtime_error);
   EXPECT_THROW((PIsing{Infinite{3}, {{3, 3.0}}}.CalculateEnergy({-1,+1,+1,+1})), std::runtime_error);
}


}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_MODEL_CLASSICAL_POLYNOMIAL_ISING_HPP_ */
