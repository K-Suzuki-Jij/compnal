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

#pragma once

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
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 1, 1}));
   EXPECT_EQ(pising.GetSpinScaleFactor(), 1);
   pising.SetSpinMagnitude(1.5, 1);
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 3, 1, 1}));
   EXPECT_THROW(pising.SetSpinMagnitude(1.499, 1), std::invalid_argument);
   EXPECT_THROW(pising.SetSpinMagnitude(1.5, 10), std::invalid_argument);
   EXPECT_THROW((PIsing{chain, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}, 1, 0}), std::invalid_argument);
   
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {}        }.CalculateEnergy(std::vector<double>{-1,+1,-1})), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{0, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), -1.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{2, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), -4.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{3, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +2.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::OBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +1.0);
   
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {}        }.CalculateEnergy(std::vector<double>{-1,+1,-1})), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{0, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), -1.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{2, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), -2.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{3, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +6.0);
   EXPECT_DOUBLE_EQ((PIsing{Chain{3, BC::PBC}, {{1, 1.0}, {3, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +5.0);
   
   EXPECT_THROW((PIsing{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Chain{3, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1,+1,+1})), std::range_error);
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
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
   EXPECT_EQ(pising.GetSpinScaleFactor(), 1);
   pising.SetSpinMagnitude(1.5, {2, 0});
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
   EXPECT_THROW(pising.SetSpinMagnitude(1.499, {0, 0}), std::invalid_argument);
   EXPECT_THROW(pising.SetSpinMagnitude(1.5, {10, 0}), std::invalid_argument);
   EXPECT_THROW((PIsing{square, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}, 1, 0}), std::invalid_argument);
   
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
   
   EXPECT_THROW((PIsing{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Square{3, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1,+1,+1,+1,+1,+1})), std::range_error);
}

TEST(ModelClassical, PolynomialIsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using PIsing = model::classical::PolynomialIsing<Cubic>;
   const std::vector<typename PIsing::PHQType> spins = {
      -1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1
   };
   
   Cubic cubic{3, 2, 2, BC::PBC};
   PIsing pising{cubic, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}, 3, 3};
   
   EXPECT_EQ(pising.GetInteraction().size(), 4);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(0), -1.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(1), +2.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(2), -3.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(3), +4.0);
   EXPECT_EQ(pising.GetDegree(), 3);
   EXPECT_NO_THROW(pising.GetLattice());
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}));
   EXPECT_EQ(pising.GetSpinScaleFactor(), 3);
   pising.SetSpinMagnitude(1.5, {2, 0, 1});
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{6, 6, 6, 6, 6, 6, 6, 6, 3, 6, 6, 6}));
   EXPECT_THROW(pising.SetSpinMagnitude(1.499, {0, 0, 0}), std::invalid_argument);
   EXPECT_THROW(pising.SetSpinMagnitude(1.5, {10, 0, 0}), std::invalid_argument);
   EXPECT_THROW((PIsing{cubic, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}, 1, 0}), std::invalid_argument);
   
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
   
   EXPECT_THROW((PIsing{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Cubic{3, 2, 2, BC::PBC}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1,+1})), std::range_error);
}

TEST(ModelClassical, PolynomialIsingOnInfiniteRang) {
   using Infinite = lattice::InfiniteRange;
   using PIsing = model::classical::PolynomialIsing<Infinite>;
   
   Infinite infinite{4};
   PIsing pising{infinite, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}, 3.5, 6};
   
   EXPECT_EQ(pising.GetInteraction().size(), 4);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(0), -1.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(1), +2.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(2), -3.0);
   EXPECT_DOUBLE_EQ(pising.GetInteraction().at(3), +4.0);
   EXPECT_EQ(pising.GetDegree(), 3);
   EXPECT_NO_THROW(pising.GetLattice());
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{7, 7, 7, 7}));
   EXPECT_EQ(pising.GetSpinScaleFactor(), 6);
   pising.SetSpinMagnitude(1.5, 1);
   EXPECT_EQ(pising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{7, 3, 7, 7}));
   EXPECT_THROW(pising.SetSpinMagnitude(1.499, 0), std::invalid_argument);
   EXPECT_THROW(pising.SetSpinMagnitude(1.5, 10), std::invalid_argument);
   EXPECT_THROW((PIsing{infinite, {{0, -1.0}, {1, +2.0}, {2, -3.0}, {3, +4.0}}, 1, 0}), std::invalid_argument);
   
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{}      }}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +0.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{0, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{1, 1.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), -1.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{2, 2.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), -2.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{3, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), +3.0);
   EXPECT_DOUBLE_EQ((PIsing{Infinite{3}, {{1, 1.0}, {2, 2.0}, {3, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1,-1})), 0.0);
   
   EXPECT_THROW((PIsing{Infinite{3}, {{3, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1}))      , std::range_error);
   EXPECT_THROW((PIsing{Infinite{3}, {{3, 3.0}}}.CalculateEnergy(std::vector<double>{-1,+1,+1,+1})), std::range_error);
}


}  // namespace test
}  // namespace compnal
