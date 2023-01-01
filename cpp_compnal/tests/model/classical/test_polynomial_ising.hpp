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
//  Created by Kohei Suzuki on 2022/06/10.
//

#ifndef COMPNAL_TEST_MODEL_POLYNOMIAL_ISING_HPP_
#define COMPNAL_TEST_MODEL_POLYNOMIAL_ISING_HPP_

#include "../../../src/model/classical/polynomial_ising.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(ModelPolynomialIsing, Chain) {
   using OPType = typename model::PolynomialIsing<lattice::Chain, double>::OPType;
   auto model = model::make_polynomial_ising<lattice::Chain, double>(lattice::Chain{4}, {{1, -1.0}, {3, +2.0}});
   EXPECT_EQ(model.GetInteraction(), (std::vector<double>{0.0, -1.0, 0.0, +2.0}));
   EXPECT_EQ(model.GetSystemSize(), 4);
   EXPECT_EQ(model.GetBoundaryCondition(), lattice::BoundaryCondition::OBC);
   EXPECT_EQ(model.GetDegree(), 3);
   EXPECT_EQ(model.CalculateEnergy(std::vector<OPType>{-1, +1, +1}), 0.0);
   
}

TEST(ModelPolynomialIsing, AnyLattice) {
   auto model = model::make_polynomial_ising<lattice::AnyLattice, double>(lattice::AnyLattice{}, {{{1, 2, "a"}, -1.0}, {{1, utility::AnyTupleType{1, 1}}, -2.0}});
      
   EXPECT_EQ(model.GetSystemSize(), 4);
   EXPECT_EQ(model.GetBoundaryCondition(), lattice::BoundaryCondition::NONE);
}

TEST(ModelPolynomialIsing, InfiniteRange) {
   
   lattice::InfiniteRange lattice{10};
   std::unordered_map<int, double> interaction{{0, -1.0}, {3, +1.0}};
   auto model = model::make_polynomial_ising<lattice::InfiniteRange, double>(lattice, interaction);
   model.CalculateEnergy({});
   
}

} // namespace test
} // namespace compnal


#endif /* COMPNAL_TEST_MODEL_POLYNOMIAL_ISING_HPP_ */
