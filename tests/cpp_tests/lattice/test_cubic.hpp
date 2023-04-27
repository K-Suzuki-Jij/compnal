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
//  test_cubic.hpp
//  compnal
//
//  Created by kohei on 2023/04/28.
//  
//

#ifndef COMPNAL_TEST_LATTICE_CUBIC_HPP_
#define COMPNAL_TEST_LATTICE_CUBIC_HPP_

#include "../../../include/lattice/cubic.hpp"

namespace compnal {
namespace test {

TEST(Lattice, Cubic) {
   lattice::Cubic cubic{2, 3, 2, lattice::BoundaryCondition::OBC};
   EXPECT_EQ(cubic.GetXSize(), 2);
   EXPECT_EQ(cubic.GetYSize(), 3);
   EXPECT_EQ(cubic.GetZSize(), 2);
   EXPECT_EQ(cubic.GetSystemSize(), 12);
   EXPECT_EQ(cubic.GetBoundaryCondition(), lattice::BoundaryCondition::OBC);
   
   std::vector<std::tuple<std::int32_t, std::int32_t, std::int32_t>> coo_list{
      {0, 0, 0}, {0, 0, 1},
      {0, 1, 0}, {0, 1, 1},
      {0, 2, 0}, {0, 2, 1},
      {1, 0, 0}, {1, 0, 1},
      {1, 1, 0}, {1, 1, 1},
      {1, 2, 0}, {1, 2, 1}
   };
   
   EXPECT_EQ(cubic.GenerateCoordinateList(), coo_list);
   EXPECT_THROW((lattice::Cubic{0, 2, 1, lattice::BoundaryCondition::PBC}), std::runtime_error);
   EXPECT_THROW((lattice::Cubic{2, -1, 2, lattice::BoundaryCondition::PBC}), std::runtime_error);
   EXPECT_THROW((lattice::Cubic{2, 3, 0, lattice::BoundaryCondition::PBC}), std::runtime_error);
   EXPECT_THROW((lattice::Cubic{2, 3, 2, lattice::BoundaryCondition::NONE}), std::runtime_error);
}

}  // namespace test
}  // namespace compnal

#endif /* COMPNAL_TEST_LATTICE_CUBIC_HPP_ */
