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
//  test_chain.hpp
//  compnal
//
//  Created by kohei on 2023/04/24.
//
//

#ifndef COMPNAL_TEST_LATTICE_CHAIN_HPP_
#define COMPNAL_TEST_LATTICE_CHAIN_HPP_

#include "../../../include/lattice/chain.hpp"

namespace compnal {
namespace test {

TEST(Lattice, Chain) {
   lattice::Chain chain{3, lattice::BoundaryCondition::OBC};
   EXPECT_EQ(chain.GetSystemSize(), 3);
   EXPECT_EQ(chain.GetBoundaryCondition(), lattice::BoundaryCondition::OBC);
   EXPECT_EQ(chain.GenerateCoordinateList(),
             (std::vector<std::int32_t>{0, 1, 2}));

   EXPECT_THROW((lattice::Chain{-1, lattice::BoundaryCondition::OBC}),
                std::invalid_argument);
   EXPECT_THROW((lattice::Chain{+2, lattice::BoundaryCondition::NONE}),
                std::invalid_argument);
}

}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_LATTICE_CHAIN_HPP_ */
