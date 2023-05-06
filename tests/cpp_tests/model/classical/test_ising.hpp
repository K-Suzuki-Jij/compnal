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
//  Created by kohei on 2023/05/01.
//  
//

#ifndef COMPNAL_TEST_MODEL_CLASSICAL_ISING_HPP_
#define COMPNAL_TEST_MODEL_CLASSICAL_ISING_HPP_

#include "../../../../include/lattice/all.hpp"
#include "../../../../include/model/classical/ising.hpp"

namespace compnal {
namespace test {

TEST(ModelClassical, IsingOnChain) {
   using BC = lattice::BoundaryCondition;
   using Chain = lattice::Chain;
   using Ising = model::classical::Ising<Chain>;
   
   Chain chain_obc{3, BC::OBC};
   Ising ising_obc{chain_obc, 1.0, -4.0};
   EXPECT_DOUBLE_EQ(ising_obc.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising_obc.GetQuadratic(), -4.0);
   EXPECT_NO_THROW(ising_obc.GetLattice());
   EXPECT_EQ(ising_obc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 1}));
   EXPECT_EQ(ising_obc.GetSpinScaleFactor(), 1);
   EXPECT_DOUBLE_EQ(ising_obc.CalculateEnergy({-1, +1, -1}), 7.0);
   EXPECT_THROW(ising_obc.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising_obc.CalculateEnergy({-1, +1, +1, +1}), std::range_error);
   ising_obc.SetSpinMagnitude(1.5, 1);
   EXPECT_EQ(ising_obc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 3, 1}));
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.499, 1), std::invalid_argument);
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.5, 10), std::invalid_argument);
   EXPECT_THROW((Ising{chain_obc, 1.0, -4.0, 0.5, 0}), std::invalid_argument);
   
   Chain chain_pbc{3, BC::PBC};
   Ising ising_pbc{chain_pbc, 1.0, -4.0, 1, 2};
   EXPECT_DOUBLE_EQ(ising_pbc.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising_pbc.GetQuadratic(), -4.0);
   EXPECT_NO_THROW(ising_pbc.GetLattice());
   EXPECT_EQ(ising_pbc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{2, 2, 2}));
   EXPECT_EQ(ising_pbc.GetSpinScaleFactor(), 2);
   EXPECT_DOUBLE_EQ(ising_pbc.CalculateEnergy({-1, +1, -1}), 3.0);
   EXPECT_THROW(ising_pbc.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising_pbc.CalculateEnergy({-1, +1, +1, +1}), std::range_error);
   ising_pbc.SetSpinMagnitude(1.5, 1);
   EXPECT_EQ(ising_pbc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{2, 3, 2}));
   EXPECT_THROW(ising_pbc.SetSpinMagnitude(1.499, 1), std::invalid_argument);
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.5, 10), std::invalid_argument);
   EXPECT_THROW((Ising{chain_pbc, 1.0, -4.0, 0.5, -1}), std::invalid_argument);
}

TEST(ModelClassical, IsingOnSquare) {
   using BC = lattice::BoundaryCondition;
   using Square = lattice::Square;
   using Ising = model::classical::Ising<Square>;
   const std::vector<typename Ising::PHQType> spins = {
      -1,+1,-1,+1,+1,+1
   };
   
   Square square_obc{3, 2, BC::OBC};
   Ising ising_obc{square_obc, 1.0, 2.0};
   EXPECT_DOUBLE_EQ(ising_obc.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising_obc.GetQuadratic(), 2.0);
   EXPECT_NO_THROW(ising_obc.GetLattice());
   EXPECT_EQ(ising_obc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 1, 1, 1, 1}));
   EXPECT_EQ(ising_obc.GetSpinScaleFactor(), 1);
   EXPECT_DOUBLE_EQ(ising_obc.CalculateEnergy(spins), 0.0);
   EXPECT_THROW(ising_obc.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising_obc.CalculateEnergy({-1,+1,-1,+1,+1,+1,+1}), std::range_error);
   ising_obc.SetSpinMagnitude(1.5, {2, 0});
   EXPECT_EQ(ising_obc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 3, 1, 1, 1}));
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.499, {0, 0}), std::invalid_argument);
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.5, {10, 0}), std::invalid_argument);
   EXPECT_THROW((Ising{square_obc, 1.0, -4.0, 0.5, 0}), std::invalid_argument);
   
   Square square_pbc{3, 2, BC::PBC};
   Ising ising_pbc{square_pbc, 1.0, 2.0, 1.5, 2};
   EXPECT_DOUBLE_EQ(ising_pbc.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising_pbc.GetQuadratic(), 2.0);
   EXPECT_NO_THROW(ising_pbc.GetLattice());
   EXPECT_EQ(ising_pbc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{3, 3, 3, 3, 3, 3}));
   EXPECT_EQ(ising_pbc.GetSpinScaleFactor(), 2);
   EXPECT_DOUBLE_EQ(ising_pbc.CalculateEnergy(spins), 2.0);
   EXPECT_THROW(ising_pbc.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising_pbc.CalculateEnergy({-1,+1,-1,+1,+1,+1,+1}), std::range_error);
   ising_pbc.SetSpinMagnitude(2, {2, 0});
   EXPECT_EQ(ising_pbc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{3, 3, 4, 3, 3, 3}));
   EXPECT_THROW(ising_pbc.SetSpinMagnitude(1.499, {0, 0}), std::invalid_argument);
   EXPECT_THROW(ising_pbc.SetSpinMagnitude(1.5, {10, 0}), std::invalid_argument);
   EXPECT_THROW((Ising{square_pbc, 1.0, -4.0, 0.5, -1}), std::invalid_argument);
}

TEST(ModelClassical, IsingOnCubic) {
   using BC = lattice::BoundaryCondition;
   using Cubic = lattice::Cubic;
   using Ising = model::classical::Ising<Cubic>;
   const std::vector<typename Ising::PHQType> spins = {
      -1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1
   };
   
   Cubic cubic_obc{3, 2, 2, BC::OBC};
   Ising ising_obc{cubic_obc, 1.0, 2.0};
   EXPECT_DOUBLE_EQ(ising_obc.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising_obc.GetQuadratic(), 2.0);
   EXPECT_NO_THROW(ising_obc.GetLattice());
   EXPECT_EQ(ising_obc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
   EXPECT_EQ(ising_obc.GetSpinScaleFactor(), 1);
   EXPECT_DOUBLE_EQ(ising_obc.CalculateEnergy(spins), 12.0);
   EXPECT_THROW(ising_obc.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising_obc.CalculateEnergy({-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1,+1}), std::range_error);
   ising_obc.SetSpinMagnitude(1.5, {2, 0, 1});
   EXPECT_EQ(ising_obc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1}));
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.499, {0, 0, 0}), std::invalid_argument);
   EXPECT_THROW(ising_obc.SetSpinMagnitude(1.5, {10, 0, 0}), std::invalid_argument);
   EXPECT_THROW((Ising{cubic_obc, 1.0, -4.0, 0.5, 0}), std::invalid_argument);
   
   Cubic cubic_pbc{3, 2, 2, BC::PBC};
   Ising ising_pbc{cubic_pbc, 1.0, 2.0, 4, 2};
   EXPECT_DOUBLE_EQ(ising_pbc.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising_pbc.GetQuadratic(), 2.0);
   EXPECT_NO_THROW(ising_pbc.GetLattice());
   EXPECT_EQ(ising_pbc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8}));
   EXPECT_EQ(ising_pbc.GetSpinScaleFactor(), 2);
   EXPECT_DOUBLE_EQ(ising_pbc.CalculateEnergy(spins), 28.0);
   EXPECT_THROW(ising_pbc.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising_pbc.CalculateEnergy({-1,+1,-1,+1,+1,+1,-1,+1,-1,+1,+1,+1,+1}), std::range_error);
   ising_pbc.SetSpinMagnitude(1.5, {2, 0, 1});
   EXPECT_EQ(ising_pbc.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8}));
   EXPECT_THROW(ising_pbc.SetSpinMagnitude(1.499, {0, 0, 0}), std::invalid_argument);
   EXPECT_THROW(ising_pbc.SetSpinMagnitude(1.5, {10, 0, 0}), std::invalid_argument);
   EXPECT_THROW((Ising{cubic_pbc, 1.0, -4.0, 0.5, -1}), std::invalid_argument);
}

TEST(ModelClassical, IsingOnInfiniteRange) {
   using Infinite = lattice::InfiniteRange;
   using Ising = model::classical::Ising<Infinite>;
   
   Infinite infinite{3};
   Ising ising{infinite, 1.0, 2.0, 3.5};
   EXPECT_DOUBLE_EQ(ising.GetLinear(), 1.0);
   EXPECT_DOUBLE_EQ(ising.GetQuadratic(), 2.0);
   EXPECT_NO_THROW(ising.GetLattice());
   EXPECT_EQ(ising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{7, 7, 7}));
   EXPECT_EQ(ising.GetSpinScaleFactor(), 1);
   EXPECT_DOUBLE_EQ(ising.CalculateEnergy({-1, +1, -1}), -3.0);
   EXPECT_THROW(ising.CalculateEnergy({-1, +1}), std::range_error);
   EXPECT_THROW(ising.CalculateEnergy({-1, +1, -1, +1}), std::range_error);
   ising.SetSpinMagnitude(1.5, 1);
   EXPECT_EQ(ising.GetTwiceSpinMagnitude(), (std::vector<std::int32_t>{7, 3, 7}));
   EXPECT_THROW(ising.SetSpinMagnitude(1.499, 0), std::invalid_argument);
   EXPECT_THROW(ising.SetSpinMagnitude(1.5, 10), std::invalid_argument);
   EXPECT_THROW((Ising{infinite, 1.0, -4.0, 0.5, 0}), std::invalid_argument);
}

}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_MODEL_CLASSICAL_ISING_HPP_ */
