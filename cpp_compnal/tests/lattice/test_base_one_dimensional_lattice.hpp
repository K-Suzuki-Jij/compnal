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
//  test_base_one_dimensional_lattice.hpp
//  compnal
//
//  Created by kohei on 2022/08/11.
//  
//

#ifndef COMPNAL_TEST_LATTICE_ONE_DIMENSIONAL_LATTICE_HPP_
#define COMPNAL_TEST_LATTICE_ONE_DIMENSIONAL_LATTICE_HPP_

#include "../../src/lattice/base_one_dimensional_lattice.hpp"
#include "../../src/lattice/chain.hpp"
#include <gtest/gtest.h>

namespace compnal {
namespace test {

TEST(LatticeBaseOneDimensionalLattice, Basic) {
   
   EXPECT_EQ(lattice::BaseOneDimensionalLattice{8}.GetSystemSize(), 8);
   EXPECT_EQ(lattice::BaseOneDimensionalLattice{8}.GetBoundaryCondition(), lattice::BoundaryCondition::OBC);

   EXPECT_EQ((lattice::BaseOneDimensionalLattice{
      8, lattice::BoundaryCondition::PBC
   }.GetSystemSize()), 8);
   EXPECT_EQ((lattice::BaseOneDimensionalLattice{
      8, lattice::BoundaryCondition::PBC
   }.GetBoundaryCondition()), lattice::BoundaryCondition::PBC);

   EXPECT_THROW(lattice::BaseOneDimensionalLattice{-1}, std::runtime_error);
   EXPECT_THROW((lattice::BaseOneDimensionalLattice{1, lattice::BoundaryCondition::NONE}), std::runtime_error);

}

TEST(LatticeChain, Basic) {
   EXPECT_EQ((lattice::Chain{8}.GenerateIndexList()), (std::vector<std::int32_t>{0,1,2,3,4,5,6,7}));
}

} // namespace test
} // namespace compnal

#endif /* COMPNAL_TEST_LATTICE_ONE_DIMENSIONAL_LATTICE_HPP_ */
