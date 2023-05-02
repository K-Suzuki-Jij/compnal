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
//  test_square.hpp
//  compnal
//
//  Created by kohei on 2023/04/28.
//  
//

#ifndef COMPNAL_TEST_LATTICE_SQUARE_HPP_
#define COMPNAL_TEST_LATTICE_SQUARE_HPP_

#include "../../../include/lattice/square.hpp"

namespace compnal {
namespace test {

TEST(Lattice, Square) {
   lattice::Square square{2, 3, lattice::BoundaryCondition::OBC};
   EXPECT_EQ(square.GetXSize(), 2);
   EXPECT_EQ(square.GetYSize(), 3);
   EXPECT_EQ(square.GetSystemSize(), 6);
   EXPECT_EQ(square.GetBoundaryCondition(), lattice::BoundaryCondition::OBC);
   EXPECT_EQ(square.GenerateCoordinateList(),
             (std::vector<std::pair<std::int32_t, std::int32_t>>{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}}));
   
   EXPECT_THROW((lattice::Square{0, 2, lattice::BoundaryCondition::PBC}), std::invalid_argument);
   EXPECT_THROW((lattice::Square{2, -1, lattice::BoundaryCondition::PBC}), std::invalid_argument);
   EXPECT_THROW((lattice::Square{2, 3, lattice::BoundaryCondition::NONE}), std::invalid_argument);
   
}

}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_LATTICE_SQUARE_HPP_ */
